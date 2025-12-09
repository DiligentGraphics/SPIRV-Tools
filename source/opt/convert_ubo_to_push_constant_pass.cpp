// Copyright (c) 2025 Diligent Graphics LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/opt/convert_ubo_to_push_constant_pass.h"

#include "source/opt/ir_context.h"
#include "source/opt/type_manager.h"
#include "source/opt/decoration_manager.h"

namespace spvtools {
namespace opt {

Pass::Status ConvertUBOToPushConstantPass::Process() {
  bool modified = false;

  // Find the ID that matches the block name by searching OpName instructions
  // This could be either a variable ID or a type ID (struct type)
  uint32_t named_id = 0;
  for (auto& debug_inst : context()->module()->debugs2()) {
    if (debug_inst.opcode() == spv::Op::OpName &&
        debug_inst.GetOperand(1).AsString() == block_name_) {
      named_id = debug_inst.GetOperand(0).AsId();
      break;
    }
  }

  if (named_id == 0) {
    // Block name not found
    return Status::SuccessWithoutChange;
  }

  // Check if the named_id is a variable or a type
  Instruction* target_var = nullptr;
  Instruction* named_inst = get_def_use_mgr()->GetDef(named_id);
  
  if (named_inst == nullptr) {
    return Status::SuccessWithoutChange;
  }

  if (named_inst->opcode() == spv::Op::OpVariable) {
    // The name refers directly to a variable
    target_var = named_inst;
  } else if (named_inst->opcode() == spv::Op::OpTypeStruct) {
    // The name refers to a struct type, we need to find the variable
    // that uses a pointer to this struct type with Uniform storage class
    uint32_t struct_type_id = named_id;
    
    // Search for a variable that points to this struct type with Uniform storage class
    for (auto& inst : context()->types_values()) {
      if (inst.opcode() != spv::Op::OpVariable) {
        continue;
      }
      
      // Get the pointer type of this variable
      Instruction* ptr_type = get_def_use_mgr()->GetDef(inst.type_id());
      if (ptr_type == nullptr || ptr_type->opcode() != spv::Op::OpTypePointer) {
        continue;
      }
      
      // Check storage class is Uniform
      spv::StorageClass sc = static_cast<spv::StorageClass>(
          ptr_type->GetSingleWordInOperand(0));
      if (sc != spv::StorageClass::Uniform) {
        continue;
      }
      
      // Check if the pointee type is our struct type
      uint32_t pointee_type_id = ptr_type->GetSingleWordInOperand(1);
      if (pointee_type_id == struct_type_id) {
        target_var = &inst;
        break;
      }
    }
  }

  if (target_var == nullptr) {
    // Variable not found
    return Status::SuccessWithoutChange;
  }
  
  uint32_t target_var_id = target_var->result_id();

  // Get the pointer type of the variable
  Instruction* ptr_type_inst = get_def_use_mgr()->GetDef(target_var->type_id());
  if (ptr_type_inst == nullptr || ptr_type_inst->opcode() != spv::Op::OpTypePointer) {
    return Status::SuccessWithoutChange;
  }

  // Check if the storage class is Uniform
  spv::StorageClass storage_class =
      static_cast<spv::StorageClass>(ptr_type_inst->GetSingleWordInOperand(0));
  if (storage_class != spv::StorageClass::Uniform) {
    // Not a uniform buffer, nothing to do
    return Status::SuccessWithoutChange;
  }

  // Get the pointee type ID
  uint32_t pointee_type_id = ptr_type_inst->GetSingleWordInOperand(1);

  // Create or find a pointer type with PushConstant storage class
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  uint32_t new_ptr_type_id =
      type_mgr->FindPointerToType(pointee_type_id, spv::StorageClass::PushConstant);

  if (new_ptr_type_id == 0) {
    // Failed to create new pointer type
    return Status::Failure;
  }

  // Ensure the new pointer type is defined before the variable
  // FindPointerToType may have created it at the end, we need to move it
  Instruction* new_ptr_type_inst = get_def_use_mgr()->GetDef(new_ptr_type_id);
  if (new_ptr_type_inst != nullptr) {
    // Find the pointee type instruction to insert after it
    Instruction* pointee_type_inst = get_def_use_mgr()->GetDef(pointee_type_id);
    
    // Check if new_ptr_type_inst is after target_var in the types_values list
    bool needs_move = false;
    for (auto& inst : context()->types_values()) {
      if (&inst == target_var) {
        // Found target_var first, so new_ptr_type_inst is after it
        needs_move = true;
        break;
      }
      if (&inst == new_ptr_type_inst) {
        // Found new_ptr_type_inst first, it's in the right position
        needs_move = false;
        break;
      }
    }
    
    if (needs_move && pointee_type_inst != nullptr) {
      // Move the new pointer type to right after the pointee type
      // InsertAfter will automatically remove it from its current position
      new_ptr_type_inst->InsertAfter(pointee_type_inst);
    }
  }

  // Update the variable's type to the new pointer type
  target_var->SetResultType(new_ptr_type_id);
  
  // Also update the storage class operand of OpVariable itself
  // OpVariable has the storage class as the first operand (index 0)
  target_var->SetInOperand(0, {static_cast<uint32_t>(spv::StorageClass::PushConstant)});
  
  context()->UpdateDefUse(target_var);
  modified = true;

  // Propagate storage class change to all users of this variable
  std::set<uint32_t> seen;
  std::vector<Instruction*> users;
  get_def_use_mgr()->ForEachUser(target_var, [&users](Instruction* user) {
    users.push_back(user);
  });

  for (Instruction* user : users) {
    modified |= PropagateStorageClass(user, &seen);
  }

  // Remove Binding and DescriptorSet decorations from the variable
  auto* deco_mgr = context()->get_decoration_mgr();
  deco_mgr->RemoveDecorationsFrom(target_var_id, [](const Instruction& inst) {
    if (inst.opcode() != spv::Op::OpDecorate) {
      return false;
    }
    spv::Decoration decoration =
        static_cast<spv::Decoration>(inst.GetSingleWordInOperand(1));
    return decoration == spv::Decoration::Binding ||
           decoration == spv::Decoration::DescriptorSet;
  });

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool ConvertUBOToPushConstantPass::PropagateStorageClass(
    Instruction* inst, std::set<uint32_t>* seen) {
  if (!IsPointerResultType(inst)) {
    return false;
  }

  // Already has the correct storage class
  if (IsPointerToStorageClass(inst, spv::StorageClass::PushConstant)) {
    if (inst->opcode() == spv::Op::OpPhi) {
      if (!seen->insert(inst->result_id()).second) {
        return false;
      }
    }

    bool modified = false;
    std::vector<Instruction*> users;
    get_def_use_mgr()->ForEachUser(inst, [&users](Instruction* user) {
      users.push_back(user);
    });
    for (Instruction* user : users) {
      modified |= PropagateStorageClass(user, seen);
    }

    if (inst->opcode() == spv::Op::OpPhi) {
      seen->erase(inst->result_id());
    }
    return modified;
  }

  // Handle instructions that produce pointer results
  switch (inst->opcode()) {
    case spv::Op::OpAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
    case spv::Op::OpCopyObject:
    case spv::Op::OpPhi:
    case spv::Op::OpSelect:
      ChangeResultStorageClass(inst);
      {
        std::vector<Instruction*> users;
        get_def_use_mgr()->ForEachUser(inst, [&users](Instruction* user) {
          users.push_back(user);
        });
        for (Instruction* user : users) {
          PropagateStorageClass(user, seen);
        }
      }
      return true;

    case spv::Op::OpLoad:
    case spv::Op::OpStore:
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
      // These don't produce pointer results that need updating
      return false;

    default:
      return false;
  }
}

void ConvertUBOToPushConstantPass::ChangeResultStorageClass(Instruction* inst) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  Instruction* result_type_inst = get_def_use_mgr()->GetDef(inst->type_id());

  if (result_type_inst->opcode() != spv::Op::OpTypePointer) {
    return;
  }

  uint32_t pointee_type_id = result_type_inst->GetSingleWordInOperand(1);
  uint32_t new_result_type_id =
      type_mgr->FindPointerToType(pointee_type_id, spv::StorageClass::PushConstant);

  inst->SetResultType(new_result_type_id);
  context()->UpdateDefUse(inst);
}

bool ConvertUBOToPushConstantPass::IsPointerResultType(Instruction* inst) {
  if (inst->type_id() == 0) {
    return false;
  }

  Instruction* type_def = get_def_use_mgr()->GetDef(inst->type_id());
  return type_def != nullptr && type_def->opcode() == spv::Op::OpTypePointer;
}

bool ConvertUBOToPushConstantPass::IsPointerToStorageClass(
    Instruction* inst, spv::StorageClass storage_class) {
  if (inst->type_id() == 0) {
    return false;
  }

  Instruction* type_def = get_def_use_mgr()->GetDef(inst->type_id());
  if (type_def == nullptr || type_def->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  spv::StorageClass pointer_storage_class =
      static_cast<spv::StorageClass>(type_def->GetSingleWordInOperand(0));
  return pointer_storage_class == storage_class;
}

}  // namespace opt
}  // namespace spvtools

