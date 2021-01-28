//===- ConvertQuantumToStandard.cpp -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/ZX/ZXDialect.h"
#include "Dialect/ZX/ZXOps.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::ZX;

// Pattern rewriter
class ZXRewritePass : public ZXRewritePassBase<ZXRewritePass> {
  void runOnFunction() override;
};

// source: https://mlir.llvm.org/docs/PatternRewriter/

// Generic Rewrite Pattern for ZX Ops
template <typename MyOp>
class ZXRewritePattern : public RewritePattern {
public:
  ZXRewritePattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
  // LogicalResult match(Operation *op) const override {
  //   // The `match` method returns `success()` if the pattern is a match,
  //   // failure otherwise.
  //   // ...
  // }
  // void rewrite(Operation *op, PatternRewriter &rewriter) const override {
  //   // The `rewrite` method performs mutations on the IR rooted at `op` using
  //   // the provided rewriter. All mutations must go through the provided
  //   // rewriter.
  // }
  /// In this section, the `match` and `rewrite` implementation is specified
  /// using a single hook.
  // LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) {
  //   // The `matchAndRewrite` method performs both the matching and the
  //   // mutation.
  //   // Note that the match must reach a successful point before IR mutation
  //   // may take place.
  // }
};

//======================== ZX Rewrite Rules ===============================//

/// Rules 1, 2: Z/X Spider Fusion
template <typename NodeOp>
class ZXSpiderFusionPattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    NodeOp rootNodeOp = cast<NodeOp>(op);
    Value middleWire;
    NodeOp childNodeOp;

    bool matched = false;
    for (auto input : llvm::enumerate(rootNodeOp.getInputWires())) {
      if ((childNodeOp = input.value().template getDefiningOp<NodeOp>())) {
        middleWire = input.value();
        matched = true;
        break;
      }
    }
    if (!matched)
      return failure();

    auto combinedAngle = rewriter.create<AddFOp>(
        rewriter.getUnknownLoc(), rootNodeOp.getParam().getType(),
        childNodeOp.getParam(), rootNodeOp.getParam());

    /// angle, childNodeInputs..., rootNodeInputs...
    SmallVector<Value, 10> combinedInputs;
    combinedInputs.push_back(combinedAngle);
    combinedInputs.append(childNodeOp.getInputWires().begin(),
                          childNodeOp.getInputWires().end());
    for (auto input : rootNodeOp.getInputWires()) {
      if (input != middleWire) {
        combinedInputs.push_back(input);
      }
    }

    SmallVector<Type, 10> combinedOutputTypes(
        (childNodeOp.getNumResults() + rootNodeOp.getNumResults()) - 1,
        rewriter.getType<ZX::WireType>());

    NodeOp combinedNodeOp = rewriter.create<NodeOp>(
        rewriter.getUnknownLoc(), combinedOutputTypes, combinedInputs);
    /// childNodeOutputs..., rootNodeOutputs...
    ResultRange combinedOutputs = combinedNodeOp.getResults();

    auto outputIt = combinedOutputs.begin();
    for (Value output : childNodeOp.getResults()) {
      if (output != middleWire) {
        output.replaceAllUsesWith(*outputIt);
        ++outputIt;
      }
    }
    for (Value output : rootNodeOp.getResults()) {
      output.replaceAllUsesWith(*outputIt);
      ++outputIt;
    }
    rewriter.eraseOp(rootNodeOp);
    rewriter.eraseOp(childNodeOp);

    return success();
  }
};

/// Hadamard Color Change
/// --H--Z--H-- = ----X----
template <typename NodeOp, typename NewNodeOp>
class ZXHadamardColorChangePattern : public ZXRewritePattern<NodeOp> {
public:
  using ZXRewritePattern<NodeOp>::ZXRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // TODO: Implement.
    return failure();
    NodeOp rootNodeOp = cast<NodeOp>(op);

    bool matched = false;
    for (auto input : rootNodeOp.getInputWires()) {
      if (!input.template getDefiningOp<HOp>()) {
        matched = false;
        break;
      }
    }
    Value r;
    r.getUsers();
    for (auto output : rootNodeOp.getResults()) {
      (void)output;
    }
    if (!matched)
      return failure();

    // auto combinedAngle = rewriter.create<AddFOp>(
    //     rewriter.getUnknownLoc(), rootNodeOp.getParam().getType(),
    //     childNodeOp.getParam(), rootNodeOp.getParam());

    // /// angle, childNodeInputs..., rootNodeInputs...
    // SmallVector<Value, 10> combinedInputs;
    // combinedInputs.push_back(combinedAngle);
    // combinedInputs.append(childNodeOp.getInputWires().begin(),
    //                       childNodeOp.getInputWires().end());
    // for (auto input : rootNodeOp.getInputWires()) {
    //   if (input != middleWire) {
    //     combinedInputs.push_back(input);
    //   }
    // }

    // SmallVector<Type, 10> combinedOutputTypes(
    //     (childNodeOp.getNumResults() + rootNodeOp.getNumResults()) - 1,
    //     rewriter.getType<ZX::WireType>());

    // NodeOp combinedNodeOp = rewriter.create<NodeOp>(
    //     rewriter.getUnknownLoc(), combinedOutputTypes, combinedInputs);
    // /// childNodeOutputs..., rootNodeOutputs...
    // ResultRange combinedOutputs = combinedNodeOp.getResults();

    // auto outputIt = combinedOutputs.begin();
    // for (Value output : childNodeOp.getResults()) {
    //   if (output != middleWire) {
    //     output.replaceAllUsesWith(*outputIt);
    //     ++outputIt;
    //   }
    // }
    // for (Value output : rootNodeOp.getResults()) {
    //   output.replaceAllUsesWith(*outputIt);
    //   ++outputIt;
    // }
    // rewriter.eraseOp(rootNodeOp);
    // rewriter.eraseOp(childNodeOp);

    return success();
  }
};

/// Populate the pattern list.
void collectZXRewritePatterns(OwningRewritePatternList &patterns,
                              MLIRContext *ctx) {
  patterns.insert<ZXSpiderFusionPattern<ZOp>>(1, ctx);
  patterns.insert<ZXSpiderFusionPattern<XOp>>(1, ctx);
  patterns.insert<ZXHadamardColorChangePattern<ZOp, XOp>>(1, ctx);
  patterns.insert<ZXHadamardColorChangePattern<XOp, ZOp>>(1, ctx);
}

void ZXRewritePass::runOnFunction() {
  FuncOp func = getFunction();

  OwningRewritePatternList patterns;
  collectZXRewritePatterns(patterns, &getContext());

  applyPatternsAndFoldGreedily(func, std::move(patterns));
}

namespace mlir {

std::unique_ptr<FunctionPass> createTransformZXRewritePass() {
  return std::make_unique<ZXRewritePass>();
}

} // namespace mlir
