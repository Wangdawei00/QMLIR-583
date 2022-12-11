#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "../PassDetail.h"
#include "Dialect/Quantum/QuantumOps.h"

using namespace mlir;
using namespace mlir::quantum;

class ToyPass : public ToyPassBase<ToyPass> {
  void runOnFunction() override;
};


void ToyPass::runOnFunction() {
  auto f = getFunction();
  
}

namespace mlir {

std::unique_ptr<Pass> createToyPass() {
  return std::make_unique<ToyPass>();
}

} // namespace mlir

