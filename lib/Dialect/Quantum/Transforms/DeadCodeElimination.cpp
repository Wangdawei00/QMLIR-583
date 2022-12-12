#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/Passes.h"
#include "../PassDetail.h"
#include "Dialect/Quantum/QuantumOps.h"


namespace mlir {
#define GEN_PASS_DEF_SYMBOLDCE
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::quantum;

class DeadCodeEliminationPass : public DeadCodeEliminationPassBase<DeadCodeEliminationPass> {
  void runOnOperation() override;
  LogicalResult computeLiveness(Operation *symbolTableOp,
                                SymbolTableCollection &symbolTable,
                                bool symbolTableIsHidden,
                                DenseSet<Operation *> &liveSymbols);
};



/// Compute the liveness of the symbols within the given symbol table.
/// `symbolTableIsHidden` is true if this symbol table is known to be
/// unaccessible from operations in its parent regions.
LogicalResult DeadCodeEliminationPass::computeLiveness(Operation *symbolTableOp,
                                         SymbolTableCollection &symbolTable,
                                         bool symbolTableIsHidden,
                                         DenseSet<Operation *> &liveSymbols) {
  // A worklist of live operations to propagate uses from.
  SmallVector<Operation *, 16> worklist;

  // Walk the symbols within the current symbol table, marking the symbols that
  // are known to be live.
  for (auto &block : symbolTableOp->getRegion(0)) {
    // Add all non-symbols or symbols that can't be discarded.
    for (Operation &op : block) {
      SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(&op);
      if (!symbol) {
        worklist.push_back(&op);
        continue;
      }
      bool isDiscardable = (symbolTableIsHidden || symbol.isPrivate()) &&
                           symbol.canDiscardOnUseEmpty();
      if (!isDiscardable && liveSymbols.insert(&op).second)
        worklist.push_back(&op);
    }
  }

  // Process the set of symbols that were known to be live, adding new symbols
  // that are referenced within.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // If this is a symbol table, recursively compute its liveness.
    if (op->hasTrait<OpTrait::SymbolTable>()) {
      // The internal symbol table is hidden if the parent is, if its not a
      // symbol, or if it is a private symbol.
      SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op);
      bool symIsHidden = symbolTableIsHidden || !symbol || symbol.isPrivate();
      if (failed(computeLiveness(op, symbolTable, symIsHidden, liveSymbols)))
        return failure();
    }

    // Collect the uses held by this operation.
    Optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(op);
    if (!uses) {
      return op->emitError()
             << "operation contains potentially unknown symbol table, "
                "meaning that we can't reliable compute symbol uses";
    }

    SmallVector<Operation *, 4> resolvedSymbols;
    for (const SymbolTable::SymbolUse &use : *uses) {
      // Lookup the symbols referenced by this use.
      resolvedSymbols.clear();
      if (failed(symbolTable.lookupSymbolIn(
              op->getParentOp(), use.getSymbolRef(), resolvedSymbols)))
        // Ignore references to unknown symbols.
        continue;

      // Mark each of the resolved symbols as live.
      for (Operation *resolvedSymbol : resolvedSymbols)
        if (liveSymbols.insert(resolvedSymbol).second)
          worklist.push_back(resolvedSymbol);
    }
  }

  return success();
}

void DeadCodeEliminationPass::runOnOperation() {
  
  Operation *symbolTableOp = getOperation();

  // SymbolDCE should only be run on operations that define a symbol table.
  if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>()) {
    symbolTableOp->emitOpError()
        << " was scheduled to run under SymbolDCE, but does not define a "
           "symbol table";
    return signalPassFailure();
  }

  // A flag that signals if the top level symbol table is hidden, i.e. not
  // accessible from parent scopes.
  bool symbolTableIsHidden = true;
  SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(symbolTableOp);
  if (symbolTableOp->getParentOp() && symbol)
    symbolTableIsHidden = symbol.isPrivate();

  // Compute the set of live symbols within the symbol table.
  DenseSet<Operation *> liveSymbols;
  SymbolTableCollection symbolTable;
  if (failed(computeLiveness(symbolTableOp, symbolTable, symbolTableIsHidden,
                             liveSymbols)))
    return signalPassFailure();

  // After computing the liveness, delete all of the symbols that were found to
  // be dead.
  symbolTableOp->walk([&](Operation *nestedSymbolTable) {
    if (!nestedSymbolTable->hasTrait<OpTrait::SymbolTable>())
      return;
    for (auto &block : nestedSymbolTable->getRegion(0)) {
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (isa<SymbolOpInterface>(&op) && !liveSymbols.count(&op)) {
          op.erase();
        }
      }
    }
  });
}


namespace mlir {

std::unique_ptr<Pass> createDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}

} // namespace mlir
