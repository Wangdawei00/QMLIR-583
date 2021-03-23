//===- QuantumDialect.cpp - Quantum dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Dialect/Quantum/QuantumDialect.h"
#include "Dialect/Quantum/QuantumOps.h"
#include "Dialect/Quantum/QuantumTypes.h"
#include "TypeDetail.h"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Quantum dialect.
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with qssa
/// operations.
struct QuantumInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Quantum/QuantumOps.cpp.inc"
      >();
  addTypes<QubitType>();
  addInterfaces<QuantumInlinerInterface>();
}

Type QuantumDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef keyword;

  if (failed(parser.parseKeyword(&keyword))) {
    parser.emitError(parser.getNameLoc(), "expected type identifier");
    return Type();
  }

  // Qubit type
  if (keyword == getQubitTypeName()) {
    if (failed(parser.parseLess())) {
      parser.emitError(parser.getNameLoc(), "expected `<`");
      return Type();
    }

    uint64_t size = -1;
    if (!parser.parseOptionalInteger<uint64_t>(size).hasValue() &&
        failed(parser.parseOptionalQuestion())) {
      parser.emitError(parser.getNameLoc(), "expected an integer size or `?`");
      return Type();
    }

    if (failed(parser.parseGreater())) {
      parser.emitError(parser.getNameLoc(), "expected `>`");
      return Type();
    }

    return QubitType::get(parser.getBuilder().getContext(), size);
  }

  parser.emitError(parser.getNameLoc(), "Quantum dialect: unknown type");
  return Type();
}

void QuantumDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (type.isa<QubitType>()) {
    QubitType qubitType = type.cast<QubitType>();

    printer << getQubitTypeName() << "<";
    if (qubitType.hasStaticSize())
      printer << qubitType.getSize();
    else
      printer << "?";
    printer << ">";

    return;
  }
}
