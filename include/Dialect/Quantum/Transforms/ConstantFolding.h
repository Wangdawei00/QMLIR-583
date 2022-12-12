#ifndef CONSTANTFOLDING_H
#define CONSTANTFOLDING_H

// #define GET_TYPEDEF_CLASSES
struct EulerAngles {
  double a1, a2, a3;
  static EulerAngles fromYZYtoZYZ(double y1, double z1, double y2);
  static Attribute wrapZYZa1(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2);
  static Attribute wrapZYZa2(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2);
  static Attribute wrapZYZa3(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2);
};
#endif