#include "QuantumToStandard/ConvertQuantumToStandard.h"
#include "QuantumToStandard/Passes.h"
#include "Quantum/QuantumDialect.h"
#include "Quantum/QuantumOps.h"
#include "Quantum/QuantumTypes.h"
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
//#include "llvm/ADT/ArrayRef.h"
//#include "llvm/ADT/None.h"
//#include "llvm/ADT/STLExtras.h"
//#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace quantum;
using namespace scf;

namespace {

//===----------------------------------------------------------------------===//
// Rewriting Pattern
//===----------------------------------------------------------------------===//

class FuncOpLowering : public QuantumOpToStdPattern<FuncOp> {
public:
  using QuantumOpToStdPattern<FuncOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = operation->getLoc();
    auto funcOp = cast<FuncOp>(operation);

    // Convert the original function arguments
    TypeConverter::SignatureConversion inputs(funcOp.getNumArguments());
    for (auto &en : llvm::enumerate(funcOp.getType().getInputs()))
      inputs.addInputs(en.index(), typeConverter.convertType(en.value()));

    TypeConverter::SignatureConversion results(funcOp.getNumArguments());
    for (auto &en : llvm::enumerate(funcOp.getType().getResults()))
      results.addInputs(en.index(), typeConverter.convertType(en.value()));

    auto funcType =
        FunctionType::get(inputs.getConvertedTypes(),
                          results.getConvertedTypes(), funcOp.getContext());

    // Replace the function by a function with an updated signature
    auto newFuncOp =
        rewriter.create<FuncOp>(loc, funcOp.getName(), funcType, llvm::None);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Convert the signature and delete the original operation
    rewriter.applySignatureConversion(&newFuncOp.getBody(), inputs);
    rewriter.eraseOp(funcOp);
    return success();
  }
};

class AllocateOpLowering : public QuantumOpToStdPattern<AllocateOp> {
public:
  using QuantumOpToStdPattern<AllocateOp>::QuantumOpToStdPattern;

  LogicalResult
  matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Find the acquire_qubits function, or declare if it doesn't exist.
//    auto module = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();
//    auto acquireFunc = module.lookupSymbol<FuncOp>("acquire_qubits");
//    if (!acquireFunc) {
//      OpBuilder::InsertionGuard guard(rewriter);
//      rewriter.setInsertionPointToStart(module.getBody());
//      acquireFunc = rewriter.create<FuncOp>(
//        rewriter.getUnknownLoc(), "acquire_qubits", FunctionType::get());
//    }

    rewriter.eraseOp(operation);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

class QuantumToStdTarget : public ConversionTarget {
public:
  explicit QuantumToStdTarget(MLIRContext &context)
      : ConversionTarget(context) {}

//  bool isDynamicallyLegal(Operation *op) const override {
//    return true;
//  }
};

//===----------------------------------------------------------------------===//
// Rewriting Pass
//===----------------------------------------------------------------------===//

struct QuantumToStandardPass
    : public QuantumToStandardPassBase<QuantumToStandardPass> {
  void runOnOperation() override;
};

void QuantumToStandardPass::runOnOperation() {
  OwningRewritePatternList patterns;
  auto module = getOperation();

  QuantumTypeConverter typeConverter(module.getContext());
  populateQuantumToStdConversionPatterns(typeConverter, patterns);

  QuantumToStdTarget target(*(module.getContext()));
  
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<SCFDialect>();
  target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
    auto funcType = funcOp.getType();
    for (auto& arg: llvm::enumerate(funcType.getInputs())) {
      if (arg.value().isa<QubitType>())
        return false;
    }
    for (auto& arg: llvm::enumerate(funcType.getResults())) {
      if (arg.value().isa<QubitType>())
        return false;
    }
    return true;
  });
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addIllegalDialect<QuantumDialect>();
  
  if (failed(applyFullConversion(module, target, patterns))) {
    signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace quantum {

// Populate the conversion pattern list
void populateQuantumToStdConversionPatterns(
    QuantumTypeConverter &typeConverter,
    mlir::OwningRewritePatternList &patterns) {
  patterns.insert<FuncOpLowering, AllocateOpLowering>(typeConverter);
}

//===----------------------------------------------------------------------===//
// Quantum Type Converter
//===----------------------------------------------------------------------===//

QuantumTypeConverter::QuantumTypeConverter(MLIRContext *context)
    : context(context) {
  // Add type conversions
  addConversion([&](QubitType qubitType) -> Type {
    return MemRefType::get(qubitType.getMemRefShape(),
                           qubitType.getMemRefType());
  });
  addConversion([&](Type type) -> Optional<Type> {
    if (type.isa<QubitType>())
      return llvm::None;
    return type;
  });
}

//===----------------------------------------------------------------------===//
// Quantum Pattern Base Class
//===----------------------------------------------------------------------===//

QuantumToStdPattern::QuantumToStdPattern(StringRef rootOpName,
                                         QuantumTypeConverter &typeConverter,
                                         PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter.getContext()),
      typeConverter(typeConverter) {}

} // namespace quantum
} // namespace mlir

std::unique_ptr<Pass> mlir::createConvertQuantumToStandardPass() {
  return std::make_unique<QuantumToStandardPass>();
}
