#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.cpp.inc"

#include "../PassDetail.h"
#include "Dialect/Quantum/QuantumOps.h"
#include <array>
#include <cmath>

using namespace mlir;
using namespace mlir::quantum;
using std::sin;
using std::cos;
using std::atan2;
using std::sqrt;

class QuantumRewritePass : public QuantumRewritePassBase<QuantumRewritePass> {
  void runOnFunction() override;
};

namespace {
#include "Dialect/Quantum/Transforms/QuantumRewrites.h"
} // namespace

typedef std::vector<std::vector<double>> RotationMatrix;


static RotationMatrix RotationY(double t) {
  RotationMatrix m = {{cos(t), 0, sin(t)}, {0, 1, 0}, {-sin(t), 0, cos(t)}};
  return m;
}
static RotationMatrix RotationZ(double t) {
  RotationMatrix m = {{cos(t), -sin(t), 0}, {sin(t), cos(t), 0}, {0, 0, 1}};
  return m;
}

RotationMatrix operator*(const RotationMatrix &m1, const RotationMatrix &m2) {
  RotationMatrix m;
  for (unsigned i = 0; i < 3; i++) {
    for (unsigned j = 0; j < 3; j++) {
      m[i][j] = 0.0;
      for (unsigned k = 0; k < 3; k++) {
        m[i][j] = m1[i][k] * m2[k][j];
      }
    }
  }
  return m;
}

RotationMatrix YZY2Rotation(double y1, double z1, double y2) {
  return RotationY(y2) * RotationZ(z1) * RotationY(y1);
};

EulerAngles Rotation2ZYZ(const RotationMatrix &m) {
  double z1 = atan2(m[2][1], -m[2][1]);
  double y1 = atan2(sqrt(1 - m[2][2] * m[2][2]), m[2][2]);
  double z2 = atan2(m[1][2], m[0][2]);
  return {z1, y1, z2};
};

EulerAngles EulerAngles::fromYZYtoZYZ(double y1, double z1, double y2) {
  auto m = YZY2Rotation(y1, z1, y2);
  return Rotation2ZYZ(m);
}

Attribute EulerAngles::wrapZYZa1(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2) {
  double dy1 = y1.cast<FloatAttr>().getValueAsDouble();
  double dz11 = z11.cast<FloatAttr>().getValueAsDouble();
  double dz12 = z12.cast<FloatAttr>().getValueAsDouble();
  double dy2 = y2.cast<FloatAttr>().getValueAsDouble();
  double dz1 = dz11 + dz12;
  auto ea = fromYZYtoZYZ(dy1, dz1, dy2);
  return builder.getF64FloatAttr(ea.a1);
}

Attribute EulerAngles::wrapZYZa2(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2) {
  double dy1 = y1.cast<FloatAttr>().getValueAsDouble();
  double dz11 = z11.cast<FloatAttr>().getValueAsDouble();
  double dz12 = z12.cast<FloatAttr>().getValueAsDouble();
  double dy2 = y2.cast<FloatAttr>().getValueAsDouble();
  double dz1 = dz11 + dz12;
  auto ea = fromYZYtoZYZ(dy1, dz1, dy2);
  return builder.getF64FloatAttr(ea.a2);
}

Attribute EulerAngles::wrapZYZa3(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2) {
  double dy1 = y1.cast<FloatAttr>().getValueAsDouble();
  double dz11 = z11.cast<FloatAttr>().getValueAsDouble();
  double dz12 = z12.cast<FloatAttr>().getValueAsDouble();
  double dy2 = y2.cast<FloatAttr>().getValueAsDouble();
  double dz1 = dz11 + dz12;
  auto ea = fromYZYtoZYZ(dy1, dz1, dy2);
  return builder.getF64FloatAttr(ea.a3);
}



// U(theta, phi, lambda)
//           = U(theta2, phi2, lambda2).U(theta1, phi1, lambda1)
//           = Rz(phi2).Ry(theta2).Rz(lambda2+phi1).Ry(theta1).Rz(lambda1)
//           = Rz(phi2).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda1)
//           = U(theta', phi2 + phi', lambda1 + lambda')
class UOpMergePattern : public OpRewritePattern<UniversalRotationGateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UniversalRotationGateOp uOp,
                                PatternRewriter &rewriter) const final {
    auto parentUOp = uOp.qinp().getDefiningOp<UniversalRotationGateOp>();
    if (!parentUOp)
      return failure();

    /// Dummy implementation for testing only
    /// TODO: implement proper merge: U.U
    rewriter.replaceOp(uOp, parentUOp.qout());

    return success();
  }
};

// %b1, %a1 = CNOT %b0, %a0
// %a2, %b2 = CNOT %a1, %b1
// -----
// %b2, %a2 = CNOT %a0, %b0
struct AlternateCNOTPattern : public OpRewritePattern<CNOTGateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CNOTGateOp op,
                                PatternRewriter &rewriter) const final {
    auto parentOp = op.qinp_cont().getDefiningOp<CNOTGateOp>();
    if (!parentOp || parentOp != op.qinp_targ().getDefiningOp<CNOTGateOp>())
      return failure();
    if (op.qinp_cont() != parentOp.qout_targ())
      return failure();
    if (op.qinp_targ() != parentOp.qout_cont())
      return failure();

    auto results = rewriter.create<CNOTGateOp>(
        op->getLoc(), parentOp.qinp_targ(), parentOp.qinp_cont());
    rewriter.replaceOp(op, {results.qout_targ(), results.qout_cont()});
    return success();
  }
};

void QuantumRewritePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.insert<UOpMergePattern, AlternateCNOTPattern>(&getContext());
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace mlir {

std::unique_ptr<FunctionPass> createQuantumRewritePass() {
  return std::make_unique<QuantumRewritePass>();
}

} // namespace mlir
