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

#ifndef SOURCE_OPT_CONVERT_UBO_TO_PUSH_CONSTANT_PASS_H_
#define SOURCE_OPT_CONVERT_UBO_TO_PUSH_CONSTANT_PASS_H_

#include <string>
#include <set>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// A pass that converts a uniform buffer variable to a push constant.
// This pass:
// 1. Finds the variable with the specified block name
// 2. Changes its storage class from Uniform to PushConstant
// 3. Updates all pointer types that reference this variable
// 4. Removes Binding and DescriptorSet decorations
class ConvertUBOToPushConstantPass : public Pass {
 public:
  explicit ConvertUBOToPushConstantPass(const std::string& block_name)
      : block_name_(block_name) {}

  const char* name() const override { return "convert-ubo-to-push-constant"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    // This pass modifies types and decorations
    return IRContext::kAnalysisNone;
  }

 private:
  // Recursively updates the storage class of pointer types used by instructions
  // that reference the target variable.
  bool PropagateStorageClass(Instruction* inst, std::set<uint32_t>* seen);

  // Changes the result type of an instruction to use the new storage class.
  void ChangeResultStorageClass(Instruction* inst);

  // Checks if the instruction result type is a pointer.
  bool IsPointerResultType(Instruction* inst);

  // Checks if the instruction result type is a pointer to the specified storage class.
  bool IsPointerToStorageClass(Instruction* inst, spv::StorageClass storage_class);

  std::string block_name_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONVERT_UBO_TO_PUSH_CONSTANT_PASS_H_

