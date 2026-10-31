/*
 * Copyright © 2026 g16
 * SPDX-License-Identifier: MIT
 */

/* SPV_EXT_long_vector and SPV_NV_cooperative_vector.
 *
 * OpTypeVectorIdEXT / OpTypeCooperativeVectorNV and OpTypeVector of a
 * component count other than 2..4 are arrays of their component type (with
 * the component size as stride, so they lay out like vectors in memory).
 * Loads, stores, phis, function parameters, constants and composite
 * extract / insert act on them as arrays.  Component-wise instructions are
 * emitted once per component with the operands replaced by that component;
 * the other vector instructions are expanded here.  Cooperative vector
 * matrix-vector products and the training accumulations are loops over the
 * matrix in memory.
 */

#include "vtn_private.h"
#include "GLSL.std.450.h"

void
vtn_make_long_vector_type(struct vtn_type *t, struct vtn_type *component,
                          unsigned count)
{
   t->base_type = vtn_base_type_array;
   t->long_vector = true;
   t->length = count;
   t->array_element = component;
   t->stride = glsl_type_is_boolean(component->type) ?
               4 : glsl_get_bit_size(component->type) / 8;
   t->type = glsl_array_type(component->type, count, t->stride);
}

static unsigned
value_components(const struct glsl_type *type)
{
   return glsl_type_is_vector_or_scalar(type) ?
          glsl_get_vector_elements(type) : glsl_get_length(type);
}

/* The components of a scalar, vector or long vector. */
static nir_def **
flatten(struct vtn_builder *b, struct vtn_ssa_value *v, unsigned *count)
{
   unsigned n = value_components(v->type);
   nir_def **c = vtn_alloc_array(b, nir_def *, n);
   for (unsigned i = 0; i < n; i++)
      c[i] = v->def ? nir_channel(&b->nb, v->def, i) : v->elems[i]->def;
   *count = n;
   return c;
}

nir_def *
vtn_long_vector_head(struct vtn_builder *b, struct vtn_ssa_value *v, unsigned max)
{
   unsigned n;
   nir_def **c = flatten(b, v, &n);
   return nir_vec(&b->nb, c, MIN2(n, max));
}

/* A value of `type` (long vector, vector or scalar) from its components. */
static struct vtn_ssa_value *
compose(struct vtn_builder *b, const struct glsl_type *type, nir_def **c)
{
   struct vtn_ssa_value *v = vtn_create_ssa_value(b, type);
   if (glsl_type_is_vector_or_scalar(type)) {
      v->def = nir_vec(&b->nb, c, glsl_get_vector_elements(type));
   } else {
      for (unsigned i = 0; i < glsl_get_length(type); i++)
         v->elems[i]->def = c[i];
   }
   return v;
}

static bool
id_is_long(struct vtn_builder *b, uint32_t id)
{
   struct vtn_value *v = vtn_untyped_value(b, id);
   switch (v->value_type) {
   case vtn_value_type_ssa:
   case vtn_value_type_constant:
   case vtn_value_type_undef:
   case vtn_value_type_type:
      return v->type && vtn_type_is_long_vector(v->type);
   default:
      return false;
   }
}

static struct vtn_type *
long_pointee(struct vtn_value *v)
{
   if (v->value_type != vtn_value_type_pointer || !v->pointer->type->pointed ||
       !vtn_type_is_long_vector(v->pointer->type->pointed))
      return NULL;
   return v->pointer->type->pointed;
}

/* Emits the instruction once per component of its long vector result (or of
 * its long vector operands when `reduce` combines the scalar results).
 * Operands from `first` on that are long vectors, or pointers to them, are
 * replaced by their component; `skip` is a literal operand position.
 */
static void
split(struct vtn_builder *b, SpvOp opcode, const uint32_t *w, unsigned count,
      unsigned first, unsigned skip, nir_op reduce)
{
   struct vtn_type *res_type = vtn_get_type(b, w[1]);
   const bool res_long = vtn_type_is_long_vector(res_type);

   uint32_t ids[16];
   struct vtn_ssa_value *full[16];
   struct vtn_pointer *ptrs[16];
   struct vtn_value saved[16];
   unsigned num = 0, n = res_long ? res_type->length : 0;
   for (unsigned k = first; k < count; k++) {
      if (k == skip)
         continue;
      uint32_t id = w[k];
      bool dup = false;
      for (unsigned j = 0; j < num; j++)
         dup |= ids[j] == id;
      if (dup)
         continue;
      struct vtn_value *v = vtn_untyped_value(b, id);
      struct vtn_type *pt;
      if (v->value_type != vtn_value_type_type && id_is_long(b, id)) {
         vtn_assert(num < ARRAY_SIZE(ids));
         ids[num] = id;
         full[num] = vtn_ssa_value(b, id);
         ptrs[num] = NULL;
         n = MAX2(n, v->type->length);
         num++;
      } else if ((pt = long_pointee(v))) {
         vtn_assert(num < ARRAY_SIZE(ids));
         ids[num] = id;
         full[num] = NULL;
         ptrs[num] = v->pointer;
         num++;
      }
   }
   vtn_assert(n > 0);

   struct vtn_value *type_val = vtn_untyped_value(b, w[1]);
   struct vtn_value *res_val = vtn_untyped_value(b, w[2]);
   struct vtn_type *saved_type = type_val->type;
   struct vtn_value saved_res = *res_val;
   struct vtn_type *elem_type = res_long ? res_type->array_element : res_type;

   nir_def **c = vtn_alloc_array(b, nir_def *, n);
   for (unsigned i = 0; i < n; i++) {
      for (unsigned j = 0; j < num; j++) {
         struct vtn_value *v = &b->values[ids[j]];
         saved[j] = *v;
         struct vtn_value e = {
            .name = saved[j].name,
            .decoration = saved[j].decoration,
         };
         if (full[j]) {
            e.value_type = vtn_value_type_ssa;
            e.type = saved[j].type->array_element;
            e.ssa = full[j]->elems[i];
         } else {
            e.value_type = vtn_value_type_pointer;
            e.pointer = vtn_pointer_element(b, ptrs[j], i);
            e.type = e.pointer->type;
         }
         *v = e;
      }
      type_val->type = elem_type;
      res_val->type = elem_type;

      b->long_vector_split = true;
      vtn_dispatch_body_instruction(b, opcode, w, count);
      b->long_vector_split = false;

      vtn_assert(res_val->value_type == vtn_value_type_ssa);
      c[i] = res_val->ssa->def;
      *res_val = saved_res;
      type_val->type = saved_type;
      for (unsigned j = 0; j < num; j++)
         b->values[ids[j]] = saved[j];
   }

   if (res_long) {
      vtn_push_ssa_value(b, w[2], compose(b, res_type->type, c));
   } else {
      nir_def *r = c[0];
      for (unsigned i = 1; i < n; i++)
         r = nir_build_alu2(&b->nb, reduce, r, c[i]);
      vtn_push_nir_ssa(b, w[2], r);
   }
}

static nir_def *
dot(struct vtn_builder *b, nir_def **x, nir_def **y, unsigned n)
{
   nir_def *r = nir_fmul(&b->nb, x[0], y[0]);
   for (unsigned i = 1; i < n; i++)
      r = nir_fadd(&b->nb, r, nir_fmul(&b->nb, x[i], y[i]));
   return r;
}

static nir_def **
operand(struct vtn_builder *b, uint32_t id, unsigned *n)
{
   return flatten(b, vtn_ssa_value(b, id), n);
}

static void
handle_glsl450(struct vtn_builder *b, enum GLSLstd450 op, const uint32_t *w,
               unsigned count)
{
   nir_builder *nb = &b->nb;
   struct vtn_type *res_type = vtn_get_type(b, w[1]);
   unsigned n;
   nir_def **x = operand(b, w[5], &n);
   nir_def **c = vtn_alloc_array(b, nir_def *, n);

   switch (op) {
   case GLSLstd450Length:
      vtn_push_nir_ssa(b, w[2], nir_fsqrt(nb, dot(b, x, x, n)));
      return;

   case GLSLstd450Distance: {
      nir_def **y = operand(b, w[6], &n);
      for (unsigned i = 0; i < n; i++)
         c[i] = nir_fsub(nb, x[i], y[i]);
      vtn_push_nir_ssa(b, w[2], nir_fsqrt(nb, dot(b, c, c, n)));
      return;
   }

   case GLSLstd450Normalize: {
      nir_def *len = nir_fsqrt(nb, dot(b, x, x, n));
      for (unsigned i = 0; i < n; i++)
         c[i] = nir_fdiv(nb, x[i], len);
      break;
   }

   case GLSLstd450FaceForward: {
      nir_def **in = operand(b, w[6], &n);
      nir_def **ref = operand(b, w[7], &n);
      nir_def *neg = nir_flt(nb, dot(b, ref, in, n),
                             nir_imm_floatN_t(nb, 0.0, x[0]->bit_size));
      for (unsigned i = 0; i < n; i++)
         c[i] = nir_bcsel(nb, neg, x[i], nir_fneg(nb, x[i]));
      break;
   }

   case GLSLstd450Reflect: {
      /* I - 2 * dot(N, I) * N */
      nir_def **nn = operand(b, w[6], &n);
      nir_def *d = nir_fmul_imm(nb, dot(b, nn, x, n), 2.0);
      for (unsigned i = 0; i < n; i++)
         c[i] = nir_fsub(nb, x[i], nir_fmul(nb, d, nn[i]));
      break;
   }

   case GLSLstd450Refract: {
      nir_def **nn = operand(b, w[6], &n);
      nir_def *eta = vtn_get_nir_ssa(b, w[7]);
      if (eta->bit_size != x[0]->bit_size)
         eta = nir_type_convert(nb, eta, nir_type_float, nir_type_float | x[0]->bit_size,
                                nir_rounding_mode_undef);
      nir_def *one = nir_imm_floatN_t(nb, 1.0, x[0]->bit_size);
      nir_def *zero = nir_imm_floatN_t(nb, 0.0, x[0]->bit_size);
      nir_def *ndi = dot(b, nn, x, n);
      /* k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I)) */
      nir_def *k = nir_fsub(nb, one, nir_fmul(nb, eta, nir_fmul(nb, eta,
                            nir_fsub(nb, one, nir_fmul(nb, ndi, ndi)))));
      nir_def *f = nir_fadd(nb, nir_fmul(nb, eta, ndi), nir_fsqrt(nb, k));
      nir_def *tir = nir_flt(nb, k, zero);
      for (unsigned i = 0; i < n; i++)
         c[i] = nir_bcsel(nb, tir, zero,
                          nir_fsub(nb, nir_fmul(nb, eta, x[i]), nir_fmul(nb, f, nn[i])));
      break;
   }

   case GLSLstd450ModfStruct:
   case GLSLstd450FrexpStruct: {
      vtn_assert(glsl_type_is_struct(res_type->type));
      struct vtn_ssa_value *r = vtn_create_ssa_value(b, res_type->type);
      for (unsigned i = 0; i < n; i++) {
         nir_def *s = x[i];
         if (op == GLSLstd450FrexpStruct) {
            r->elems[0]->elems[i]->def = nir_frexp_sig(nb, s);
            r->elems[1]->elems[i]->def = nir_frexp_exp(nb, s);
         } else {
            /* vtn_glsl450.c GLSLstd450ModfStruct */
            nir_def *inf = nir_imm_floatN_t(nb, INFINITY, s->bit_size);
            nir_def *sign_bit = nir_imm_intN_t(nb, (uint64_t)1 << (s->bit_size - 1), s->bit_size);
            nir_def *signed_zero = nir_iand(nb, s, sign_bit);
            nir_def *abs = nir_fabs(nb, s);
            r->elems[0]->elems[i]->def =
               nir_bcsel(nb, nir_ieq(nb, abs, inf), signed_zero,
                         nir_ior(nb, signed_zero, nir_ffract(nb, abs)));
            r->elems[1]->elems[i]->def = nir_ior(nb, signed_zero, nir_ffloor(nb, abs));
         }
      }
      vtn_push_ssa_value(b, w[2], r);
      return;
   }

   default:
      UNREACHABLE("not a long vector GLSL.std.450 expansion");
   }
   vtn_push_ssa_value(b, w[2], compose(b, res_type->type, c));
}

static void
handle_bitcast(struct vtn_builder *b, const uint32_t *w)
{
   struct vtn_type *type = vtn_get_type(b, w[1]);
   unsigned n, m = value_components(type->type);
   nir_def **s = operand(b, w[3], &n);
   const unsigned sb = s[0]->bit_size;
   const unsigned db = glsl_get_bit_size(glsl_without_array(type->type));
   vtn_fail_if(n * sb != m * db, "OpBitcast must keep the total number of bits");
   nir_def **c = vtn_alloc_array(b, nir_def *, m);
   for (unsigned i = 0; i < m; i++) {
      unsigned bit = i * db;
      if (sb <= db)
         c[i] = nir_extract_bits(&b->nb, s + bit / sb, db / sb, 0, 1, db);
      else
         c[i] = nir_extract_bits(&b->nb, s + bit / sb, 1, bit % sb, 1, db);
   }
   vtn_push_ssa_value(b, w[2], compose(b, type->type, c));
}

static bool
is_component_wise(SpvOp opcode, unsigned *skip)
{
   *skip = 0;
   switch (opcode) {
   case SpvOpSNegate:
   case SpvOpFNegate:
   case SpvOpNot:
   case SpvOpConvertFToU:
   case SpvOpConvertFToS:
   case SpvOpConvertSToF:
   case SpvOpConvertUToF:
   case SpvOpUConvert:
   case SpvOpSConvert:
   case SpvOpFConvert:
   case SpvOpQuantizeToF16:
   case SpvOpSatConvertSToU:
   case SpvOpSatConvertUToS:
   case SpvOpIsNan:
   case SpvOpIsInf:
   case SpvOpIsFinite:
   case SpvOpIsNormal:
   case SpvOpSignBitSet:
   case SpvOpLessOrGreater:
   case SpvOpOrdered:
   case SpvOpUnordered:
   case SpvOpIAdd:
   case SpvOpFAdd:
   case SpvOpISub:
   case SpvOpFSub:
   case SpvOpIMul:
   case SpvOpFMul:
   case SpvOpFmaKHR:
   case SpvOpUDiv:
   case SpvOpSDiv:
   case SpvOpFDiv:
   case SpvOpUMod:
   case SpvOpSRem:
   case SpvOpSMod:
   case SpvOpFRem:
   case SpvOpFMod:
   case SpvOpVectorTimesScalar:
   case SpvOpShiftRightLogical:
   case SpvOpShiftRightArithmetic:
   case SpvOpShiftLeftLogical:
   case SpvOpLogicalEqual:
   case SpvOpLogicalNotEqual:
   case SpvOpLogicalOr:
   case SpvOpLogicalAnd:
   case SpvOpLogicalNot:
   case SpvOpBitwiseOr:
   case SpvOpBitwiseXor:
   case SpvOpBitwiseAnd:
   case SpvOpIEqual:
   case SpvOpFOrdEqual:
   case SpvOpFUnordEqual:
   case SpvOpINotEqual:
   case SpvOpFOrdNotEqual:
   case SpvOpFUnordNotEqual:
   case SpvOpULessThan:
   case SpvOpSLessThan:
   case SpvOpFOrdLessThan:
   case SpvOpFUnordLessThan:
   case SpvOpUGreaterThan:
   case SpvOpSGreaterThan:
   case SpvOpFOrdGreaterThan:
   case SpvOpFUnordGreaterThan:
   case SpvOpULessThanEqual:
   case SpvOpSLessThanEqual:
   case SpvOpFOrdLessThanEqual:
   case SpvOpFUnordLessThanEqual:
   case SpvOpUGreaterThanEqual:
   case SpvOpSGreaterThanEqual:
   case SpvOpFOrdGreaterThanEqual:
   case SpvOpFUnordGreaterThanEqual:
   case SpvOpDPdx:
   case SpvOpDPdy:
   case SpvOpFwidth:
   case SpvOpDPdxFine:
   case SpvOpDPdyFine:
   case SpvOpFwidthFine:
   case SpvOpDPdxCoarse:
   case SpvOpDPdyCoarse:
   case SpvOpFwidthCoarse:
   case SpvOpBitFieldInsert:
   case SpvOpBitFieldSExtract:
   case SpvOpBitFieldUExtract:
   case SpvOpBitReverse:
   case SpvOpBitCount:
   case SpvOpUCountLeadingZerosINTEL:
   case SpvOpUCountTrailingZerosINTEL:
   case SpvOpAbsISubINTEL:
   case SpvOpAbsUSubINTEL:
   case SpvOpIAddSatINTEL:
   case SpvOpUAddSatINTEL:
   case SpvOpIAverageINTEL:
   case SpvOpUAverageINTEL:
   case SpvOpIAverageRoundedINTEL:
   case SpvOpUAverageRoundedINTEL:
   case SpvOpISubSatINTEL:
   case SpvOpUSubSatINTEL:
   case SpvOpIMul32x16INTEL:
   case SpvOpUMul32x16INTEL:
   case SpvOpSelect:
   case SpvOpGroupNonUniformBroadcast:
   case SpvOpGroupNonUniformBroadcastFirst:
   case SpvOpGroupNonUniformShuffle:
   case SpvOpGroupNonUniformShuffleXor:
   case SpvOpGroupNonUniformShuffleUp:
   case SpvOpGroupNonUniformShuffleDown:
   case SpvOpGroupNonUniformQuadBroadcast:
   case SpvOpGroupNonUniformQuadSwap:
   case SpvOpGroupNonUniformRotateKHR:
   case SpvOpGroupBroadcast:
   case SpvOpSubgroupFirstInvocationKHR:
   case SpvOpSubgroupReadInvocationKHR:
   case SpvOpSubgroupShuffleINTEL:
   case SpvOpSubgroupShuffleDownINTEL:
   case SpvOpSubgroupShuffleUpINTEL:
   case SpvOpSubgroupShuffleXorINTEL:
      return true;

   case SpvOpGroupNonUniformIAdd:
   case SpvOpGroupNonUniformFAdd:
   case SpvOpGroupNonUniformIMul:
   case SpvOpGroupNonUniformFMul:
   case SpvOpGroupNonUniformSMin:
   case SpvOpGroupNonUniformUMin:
   case SpvOpGroupNonUniformFMin:
   case SpvOpGroupNonUniformSMax:
   case SpvOpGroupNonUniformUMax:
   case SpvOpGroupNonUniformFMax:
   case SpvOpGroupNonUniformBitwiseAnd:
   case SpvOpGroupNonUniformBitwiseOr:
   case SpvOpGroupNonUniformBitwiseXor:
   case SpvOpGroupNonUniformLogicalAnd:
   case SpvOpGroupNonUniformLogicalOr:
   case SpvOpGroupNonUniformLogicalXor:
   case SpvOpGroupIAdd:
   case SpvOpGroupFAdd:
   case SpvOpGroupFMin:
   case SpvOpGroupUMin:
   case SpvOpGroupSMin:
   case SpvOpGroupFMax:
   case SpvOpGroupUMax:
   case SpvOpGroupSMax:
   case SpvOpGroupIMulKHR:
   case SpvOpGroupFMulKHR:
   case SpvOpGroupBitwiseAndKHR:
   case SpvOpGroupBitwiseOrKHR:
   case SpvOpGroupBitwiseXorKHR:
   case SpvOpGroupLogicalAndKHR:
   case SpvOpGroupLogicalOrKHR:
   case SpvOpGroupLogicalXorKHR:
   case SpvOpGroupIAddNonUniformAMD:
   case SpvOpGroupFAddNonUniformAMD:
   case SpvOpGroupFMinNonUniformAMD:
   case SpvOpGroupUMinNonUniformAMD:
   case SpvOpGroupSMinNonUniformAMD:
   case SpvOpGroupFMaxNonUniformAMD:
   case SpvOpGroupUMaxNonUniformAMD:
   case SpvOpGroupSMaxNonUniformAMD:
      /* GroupOperation literal */
      *skip = 4;
      return true;

   default:
      return false;
   }
}

bool
vtn_handle_long_vector_instruction(struct vtn_builder *b, SpvOp opcode,
                                   const uint32_t *w, unsigned count)
{
   nir_builder *nb = &b->nb;
   unsigned skip;

   switch (opcode) {
   case SpvOpExtInst:
   case SpvOpExtInstWithForwardRefsKHR: {
      struct vtn_value *set = vtn_untyped_value(b, w[3]);
      if (set->value_type != vtn_value_type_extension ||
          set->ext_handler != vtn_handle_glsl450_instruction || count < 6)
         return false;
      enum GLSLstd450 op = w[4];
      switch (op) {
      case GLSLstd450Length:
      case GLSLstd450Distance:
      case GLSLstd450ModfStruct:
      case GLSLstd450FrexpStruct:
         if (!id_is_long(b, w[5]))
            return false;
         handle_glsl450(b, op, w, count);
         return true;
      case GLSLstd450Normalize:
      case GLSLstd450FaceForward:
      case GLSLstd450Reflect:
      case GLSLstd450Refract:
         if (!id_is_long(b, w[1]))
            return false;
         handle_glsl450(b, op, w, count);
         return true;
      default:
         if (!id_is_long(b, w[1]))
            return false;
         split(b, opcode, w, count, 5, 0, nir_num_opcodes);
         return true;
      }
   }

   case SpvOpDot: {
      if (!id_is_long(b, w[3]))
         return false;
      unsigned n;
      nir_def **x = operand(b, w[3], &n), **y = operand(b, w[4], &n);
      vtn_push_nir_ssa(b, w[2], dot(b, x, y, n));
      return true;
   }

   case SpvOpAny:
   case SpvOpAll: {
      if (!id_is_long(b, w[3]))
         return false;
      unsigned n;
      nir_def **x = operand(b, w[3], &n);
      nir_def *r = x[0];
      for (unsigned i = 1; i < n; i++)
         r = opcode == SpvOpAny ? nir_ior(nb, r, x[i]) : nir_iand(nb, r, x[i]);
      vtn_push_nir_ssa(b, w[2], r);
      return true;
   }

   case SpvOpGroupNonUniformAllEqual:
   case SpvOpSubgroupAllEqualKHR: {
      unsigned v = opcode == SpvOpSubgroupAllEqualKHR ? 3 : 4;
      if (!id_is_long(b, w[v]))
         return false;
      split(b, opcode, w, count, 3, 0, nir_op_iand);
      return true;
   }

   case SpvOpVectorExtractDynamic: {
      if (!id_is_long(b, w[3]))
         return false;
      unsigned n;
      nir_def **x = operand(b, w[3], &n);
      nir_def *idx = vtn_get_nir_ssa(b, w[4]);
      nir_def *r = x[0];
      for (unsigned i = 1; i < n; i++)
         r = nir_bcsel(nb, nir_ieq_imm(nb, idx, i), x[i], r);
      vtn_push_nir_ssa(b, w[2], r);
      return true;
   }

   case SpvOpVectorInsertDynamic: {
      if (!id_is_long(b, w[1]))
         return false;
      unsigned n;
      nir_def **x = operand(b, w[3], &n);
      nir_def *v = vtn_get_nir_ssa(b, w[4]);
      nir_def *idx = vtn_get_nir_ssa(b, w[5]);
      for (unsigned i = 0; i < n; i++)
         x[i] = nir_bcsel(nb, nir_ieq_imm(nb, idx, i), v, x[i]);
      vtn_push_ssa_value(b, w[2], compose(b, vtn_get_type(b, w[1])->type, x));
      return true;
   }

   case SpvOpVectorShuffle: {
      if (!id_is_long(b, w[1]) && !id_is_long(b, w[3]) && !id_is_long(b, w[4]))
         return false;
      struct vtn_type *type = vtn_get_type(b, w[1]);
      unsigned n0, n1, m = value_components(type->type);
      nir_def **x = operand(b, w[3], &n0), **y = operand(b, w[4], &n1);
      nir_def **c = vtn_alloc_array(b, nir_def *, m);
      for (unsigned i = 0; i < m; i++) {
         uint32_t s = w[5 + i];
         vtn_fail_if(s != 0xffffffff && s >= n0 + n1,
                     "OpVectorShuffle component out of range");
         c[i] = s == 0xffffffff ? nir_undef(nb, 1, x[0]->bit_size) :
                s < n0 ? x[s] : y[s - n0];
      }
      vtn_push_ssa_value(b, w[2], compose(b, type->type, c));
      return true;
   }

   case SpvOpCompositeConstruct: {
      if (!id_is_long(b, w[1]))
         return false;
      struct vtn_type *type = vtn_get_type(b, w[1]);
      nir_def **c = vtn_alloc_array(b, nir_def *, type->length);
      unsigned k = 0;
      for (unsigned i = 3; i < count; i++) {
         unsigned n;
         nir_def **x = operand(b, w[i], &n);
         vtn_fail_if(k + n > type->length, "OpCompositeConstruct has too many components");
         memcpy(c + k, x, n * sizeof(*x));
         k += n;
      }
      vtn_fail_if(k != type->length, "OpCompositeConstruct has too few components");
      vtn_push_ssa_value(b, w[2], compose(b, type->type, c));
      return true;
   }

   case SpvOpBitcast: {
      bool rl = id_is_long(b, w[1]), sl = id_is_long(b, w[3]);
      if (!rl && !sl)
         return false;
      handle_bitcast(b, w);
      return true;
   }

   default:
      if (!is_component_wise(opcode, &skip) || !id_is_long(b, w[1]))
         return false;
      /* OpSelect of whole long vectors by a scalar condition is an array
       * select.
       */
      if (opcode == SpvOpSelect && !id_is_long(b, w[3]))
         return false;
      split(b, opcode, w, count, 3, skip, nir_num_opcodes);
      return true;
   }
}

/* SPV_NV_cooperative_vector */

enum cv_kind { CV_FLOAT, CV_SINT, CV_UINT };

struct cv_interp {
   enum cv_kind kind;
   unsigned bits;
   bool packed;
   bool e4m3, e5m2;
};

static struct cv_interp
cv_interp(struct vtn_builder *b, uint32_t id)
{
   switch (vtn_constant_uint(b, id)) {
   case SpvComponentTypeFloat16NV: return (struct cv_interp) { CV_FLOAT, 16 };
   case SpvComponentTypeFloat32NV: return (struct cv_interp) { CV_FLOAT, 32 };
   case SpvComponentTypeFloat64NV: return (struct cv_interp) { CV_FLOAT, 64 };
   case SpvComponentTypeSignedInt8NV: return (struct cv_interp) { CV_SINT, 8 };
   case SpvComponentTypeSignedInt16NV: return (struct cv_interp) { CV_SINT, 16 };
   case SpvComponentTypeSignedInt32NV: return (struct cv_interp) { CV_SINT, 32 };
   case SpvComponentTypeSignedInt64NV: return (struct cv_interp) { CV_SINT, 64 };
   case SpvComponentTypeUnsignedInt8NV: return (struct cv_interp) { CV_UINT, 8 };
   case SpvComponentTypeUnsignedInt16NV: return (struct cv_interp) { CV_UINT, 16 };
   case SpvComponentTypeUnsignedInt32NV: return (struct cv_interp) { CV_UINT, 32 };
   case SpvComponentTypeUnsignedInt64NV: return (struct cv_interp) { CV_UINT, 64 };
   case SpvComponentTypeSignedInt8PackedNV: return (struct cv_interp) { CV_SINT, 8, .packed = true };
   case SpvComponentTypeUnsignedInt8PackedNV: return (struct cv_interp) { CV_UINT, 8, .packed = true };
   case SpvComponentTypeFloatE4M3NV: return (struct cv_interp) { CV_FLOAT, 8, .e4m3 = true };
   case SpvComponentTypeFloatE5M2NV: return (struct cv_interp) { CV_FLOAT, 8, .e5m2 = true };
   default: vtn_fail("unknown cooperative vector component type");
   }
}

static const struct glsl_type *
cv_glsl_type(struct cv_interp t)
{
   if (t.kind == CV_FLOAT && t.bits > 8)
      return glsl_floatN_t_type(t.bits);
   return t.kind == CV_SINT ? glsl_intN_t_type(t.bits) : glsl_uintN_t_type(t.bits);
}

static nir_deref_instr *
cv_base(struct vtn_builder *b, uint32_t id)
{
   struct vtn_pointer *p = vtn_value_to_pointer(b, vtn_value(b, id, vtn_value_type_pointer));
   if (!p->type->pointed) {
      struct vtn_type *t = vtn_zalloc(b, struct vtn_type);
      t->base_type = vtn_base_type_scalar;
      t->type = glsl_uint8_t_type();
      t->length = 1;
      p = vtn_cast_pointer(b, p, t);
   }
   return vtn_pointer_to_deref(b, p);
}

/* The object of type `t` `byte` bytes after the start of `base`. */
static nir_deref_instr *
cv_at(struct vtn_builder *b, nir_deref_instr *base, nir_def *byte,
      const struct glsl_type *t)
{
   nir_builder *nb = &b->nb;
   unsigned size = glsl_get_bit_size(t) / 8;
   nir_deref_instr *u8 = nir_build_deref_cast(nb, &base->def, base->modes, glsl_uint8_t_type(), 1);
   nir_deref_instr *at = nir_build_deref_ptr_as_array(nb, u8, nir_u2uN(nb, byte, base->def.bit_size));
   return nir_build_deref_cast_with_alignment(nb, &at->def, base->modes, t, size, size, 0);
}

/* OCP FP8 E4M3 (bias 7, no infinities, S.1111.111 NaN) / E5M2 (the high byte
 * of a binary16) to float32.
 */
static nir_def *
cv_fp8_to_f32(nir_builder *nb, nir_def *v, bool e4m3)
{
   v = nir_u2u32(nb, v);
   if (!e4m3)
      return nir_f2f32(nb, nir_u2u16(nb, nir_ishl_imm(nb, v, 8)));
   nir_def *mag = nir_iand_imm(nb, v, 0x7f);
   nir_def *norm = nir_iadd_imm(nb, nir_ishl_imm(nb, mag, 20), (127 - 7) << 23);
   nir_def *sub = nir_fmul_imm(nb, nir_u2f32(nb, mag), 1.0 / 512.0);
   nir_def *f = nir_bcsel(nb, nir_ult_imm(nb, mag, 8), sub, norm);
   f = nir_ior(nb, f, nir_ishl_imm(nb, nir_iand_imm(nb, v, 0x80), 24));
   return nir_bcsel(nb, nir_ieq_imm(nb, mag, 0x7f), nir_imm_float(nb, NAN), f);
}

/* Memory element of interpretation `t` at `byte` as a float32 / int32. */
static nir_def *
cv_load(struct vtn_builder *b, nir_deref_instr *base, nir_def *byte, struct cv_interp t)
{
   nir_builder *nb = &b->nb;
   nir_def *v = nir_load_deref(nb, cv_at(b, base, byte, cv_glsl_type(t)));
   if (t.e4m3 || t.e5m2)
      return cv_fp8_to_f32(nb, v, t.e4m3);
   if (t.kind == CV_FLOAT)
      return nir_f2f32(nb, v);
   return t.kind == CV_SINT ? nir_i2i32(nb, v) : nir_u2u32(nb, v);
}

/* Arithmetic conversion of an input component (signed if `sign`) to
 * interpretation `t`, as float32 or int32.
 */
static nir_def *
cv_convert_input(nir_builder *nb, nir_def *x, enum glsl_base_type src, bool sign,
                 struct cv_interp t)
{
   const bool src_float = glsl_base_type_is_float(src);
   if (t.kind == CV_FLOAT) {
      if (src_float)
         return nir_f2f32(nb, x);
      return sign ? nir_i2f32(nb, x) : nir_u2f32(nb, x);
   }
   const int64_t lo = t.kind == CV_SINT ? -(1ll << (t.bits - 1)) : 0;
   const int64_t hi = t.kind == CV_SINT ? (1ll << (t.bits - 1)) - 1 : (1ll << t.bits) - 1;
   if (src_float) {
      x = nir_fround_even(nb, nir_f2f32(nb, x));
      x = nir_fmin(nb, nir_fmax(nb, x, nir_imm_float(nb, lo)), nir_imm_float(nb, hi));
      return t.kind == CV_SINT || t.bits < 32 ? nir_f2i32(nb, x) : nir_f2u32(nb, x);
   }
   if (t.bits >= 32)
      return sign ? nir_i2i32(nb, x) : nir_u2u32(nb, x);
   if (sign) {
      x = nir_i2i32(nb, x);
      return nir_imin(nb, nir_imax(nb, x, nir_imm_int(nb, lo)), nir_imm_int(nb, hi));
   }
   x = nir_u2u32(nb, x);
   return nir_umin(nb, x, nir_imm_int(nb, hi));
}

static nir_variable *
cv_array(struct vtn_builder *b, const struct glsl_type *t, nir_def **c, unsigned n,
         const char *name)
{
   nir_builder *nb = &b->nb;
   nir_variable *var = nir_local_variable_create(nb->impl, glsl_array_type(t, n, 0), name);
   nir_deref_instr *d = nir_build_deref_var(nb, var);
   for (unsigned i = 0; c && i < n; i++)
      nir_store_deref(nb, nir_build_deref_array_imm(nb, d, i), c[i], 1);
   return var;
}

static nir_def *
cv_elem(nir_builder *nb, nir_variable *var, nir_def *i)
{
   return nir_load_deref(nb, nir_build_deref_array(nb, nir_build_deref_var(nb, var), i));
}

/* for (i = 0; i < n; i++), i in `var` */
static nir_def *
cv_loop_begin(nir_builder *nb, nir_variable **var, unsigned n, nir_loop **loop)
{
   *var = nir_local_variable_create(nb->impl, glsl_uint_type(), "cv_i");
   nir_store_var(nb, *var, nir_imm_int(nb, 0), 1);
   *loop = nir_push_loop(nb);
   nir_def *i = nir_load_var(nb, *var);
   nir_break_if(nb, nir_uge_imm(nb, i, n));
   return i;
}

static void
cv_loop_end(nir_builder *nb, nir_variable *var, nir_def *i, nir_loop *loop)
{
   nir_store_var(nb, var, nir_iadd_imm(nb, i, 1), 1);
   nir_pop_loop(nb, loop);
}

/* Byte steps of the row (m) and column (k) index of an M x K matrix. */
static void
cv_matrix_steps(struct vtn_builder *b, SpvCooperativeVectorMatrixLayout layout,
                bool transpose, unsigned M, unsigned K, unsigned size, nir_def *stride,
                nir_def **sm, nir_def **sk)
{
   nir_builder *nb = &b->nb;
   bool col = false;
   switch (layout) {
   case SpvCooperativeVectorMatrixLayoutRowMajorNV:
      break;
   case SpvCooperativeVectorMatrixLayoutColumnMajorNV:
      col = true;
      break;
   case SpvCooperativeVectorMatrixLayoutInferencingOptimalNV:
   case SpvCooperativeVectorMatrixLayoutTrainingOptimalNV:
      /* The optimal layouts are row-major without padding (vkConvertCooperativeVectorMatrixNV). */
      stride = nir_imm_int(nb, (transpose ? M : K) * size);
      break;
   default:
      vtn_fail("unknown cooperative vector matrix layout");
   }
   vtn_fail_if(!stride, "MatrixStride is needed for the row- and column-major layouts");
   stride = nir_u2u32(nb, stride);
   if (col != transpose) {
      *sm = nir_imm_int(nb, size);
      *sk = stride;
   } else {
      *sm = stride;
      *sk = nir_imm_int(nb, size);
   }
}

static void
cv_matrix_mul(struct vtn_builder *b, SpvOp opcode, const uint32_t *w, unsigned count)
{
   nir_builder *nb = &b->nb;
   const bool add = opcode == SpvOpCooperativeVectorMatrixMulAddNV;
   const unsigned o = add ? 3 : 0;
   struct vtn_type *res_type = vtn_get_type(b, w[1]);
   struct vtn_type *in_type = vtn_get_value_type(b, w[3]);
   struct cv_interp in_t = cv_interp(b, w[4]);
   nir_deref_instr *matrix = cv_base(b, w[5]);
   nir_def *matrix_off = nir_u2u32(nb, vtn_get_nir_ssa(b, w[6]));
   struct cv_interp mat_t = cv_interp(b, w[7]);
   const unsigned M = vtn_constant_uint(b, w[8 + o]);
   const unsigned K = vtn_constant_uint(b, w[9 + o]);
   const SpvCooperativeVectorMatrixLayout layout = vtn_constant_uint(b, w[10 + o]);
   const bool transpose = vtn_value(b, w[11 + o], vtn_value_type_constant)->constant->values[0].b;
   nir_def *stride = count > 12 + o ? vtn_get_nir_ssa(b, w[12 + o]) : NULL;
   const uint32_t operands = count > 13 + o ? w[13 + o] : 0;
   const bool in_signed = operands & SpvCooperativeMatrixOperandsMatrixBSignedComponentsKHRMask;

   vtn_fail_if(res_type->length != M, "cooperative vector result needs M components");
   const bool fp = mat_t.kind == CV_FLOAT;
   const struct glsl_type *acc_t = fp ? glsl_float_type() : glsl_int_type();

   /* Input as K float32 / int32 values of the input interpretation. */
   unsigned n;
   nir_def **x = operand(b, w[3], &n);
   nir_def **in = vtn_alloc_array(b, nir_def *, K);
   const enum glsl_base_type in_base = glsl_get_base_type(in_type->array_element->type);
   for (unsigned k = 0; k < K; k++) {
      if (in_t.packed) {
         vtn_fail_if(k / 4 >= n, "packed cooperative vector input is too short");
         nir_def *word = nir_u2u32(nb, x[k / 4]);
         unsigned sh = 8 * (k % 4);
         in[k] = in_t.kind == CV_SINT ?
                 nir_ishr_imm(nb, nir_ishl_imm(nb, word, 24 - sh), 24) :
                 nir_iand_imm(nb, nir_ushr_imm(nb, word, sh), 0xff);
      } else {
         vtn_fail_if(k >= n, "cooperative vector input needs K components");
         in[k] = cv_convert_input(nb, x[k], in_base, in_signed, in_t);
      }
      if (fp && in[k]->bit_size == 32 && in_t.kind != CV_FLOAT)
         in[k] = in_t.kind == CV_SINT ? nir_i2f32(nb, in[k]) : nir_u2f32(nb, in[k]);
      else if (!fp && in_t.kind == CV_FLOAT)
         in[k] = nir_f2i32(nb, in[k]);
   }
   nir_variable *in_var = cv_array(b, acc_t, in, K, "cv_in");
   nir_variable *out_var = cv_array(b, acc_t, NULL, M, "cv_out");

   const unsigned size = mat_t.bits / 8;
   nir_def *sm, *sk;
   cv_matrix_steps(b, layout, transpose, M, K, size, stride, &sm, &sk);

   nir_deref_instr *bias = NULL;
   nir_def *bias_off = NULL;
   struct cv_interp bias_t = { 0 };
   if (add) {
      bias = cv_base(b, w[8]);
      bias_off = nir_u2u32(nb, vtn_get_nir_ssa(b, w[9]));
      bias_t = cv_interp(b, w[10]);
   }

   nir_variable *mv, *kv, *acc = nir_local_variable_create(nb->impl, acc_t, "cv_acc");
   nir_loop *ml, *kl;
   nir_def *m = cv_loop_begin(nb, &mv, M, &ml);
   {
      nir_def *a0 = fp ? nir_imm_float(nb, 0) : nir_imm_int(nb, 0);
      if (add) {
         nir_def *bv = cv_load(b, bias, nir_iadd(nb, bias_off, nir_imul_imm(nb, m, bias_t.bits / 8)), bias_t);
         a0 = fp ? (bias_t.kind == CV_FLOAT ? bv : nir_i2f32(nb, bv)) :
                   (bias_t.kind == CV_FLOAT ? nir_f2i32(nb, bv) : bv);
      }
      nir_store_var(nb, acc, a0, 1);
      nir_def *row = nir_iadd(nb, matrix_off, nir_imul(nb, m, sm));
      nir_def *k = cv_loop_begin(nb, &kv, K, &kl);
      {
         nir_def *e = cv_load(b, matrix, nir_iadd(nb, row, nir_imul(nb, k, sk)), mat_t);
         nir_def *xi = cv_elem(nb, in_var, k);
         nir_def *a = nir_load_var(nb, acc);
         nir_store_var(nb, acc, fp ? nir_ffma(nb, e, xi, a) : nir_iadd(nb, a, nir_imul(nb, e, xi)), 1);
      }
      cv_loop_end(nb, kv, k, kl);
      nir_store_deref(nb, nir_build_deref_array(nb, nir_build_deref_var(nb, out_var), m),
                      nir_load_var(nb, acc), 1);
   }
   cv_loop_end(nb, mv, m, ml);

   const struct glsl_type *rt = res_type->array_element->type;
   nir_def **c = vtn_alloc_array(b, nir_def *, M);
   for (unsigned i = 0; i < M; i++) {
      nir_def *r = cv_elem(nb, out_var, nir_imm_int(nb, i));
      if (glsl_base_type_is_float(glsl_get_base_type(rt)))
         c[i] = fp ? nir_f2fN(nb, r, glsl_get_bit_size(rt)) : nir_i2fN(nb, r, glsl_get_bit_size(rt));
      else
         c[i] = nir_i2iN(nb, fp ? nir_f2i32(nb, r) : r, glsl_get_bit_size(rt));
   }
   vtn_push_ssa_value(b, w[2], compose(b, res_type->type, c));
}

/* Relaxed device-scope atomic float add of `v` at `byte`; binary16 through a
 * compare-exchange loop on its 32-bit word.
 */
static void
cv_atomic_fadd(struct vtn_builder *b, nir_deref_instr *base, nir_def *byte, nir_def *v)
{
   nir_builder *nb = &b->nb;
   if (v->bit_size == 32) {
      nir_deref_atomic(nb, 32, &cv_at(b, base, byte, glsl_float_type())->def, v,
                       .atomic_op = nir_atomic_op_fadd);
      return;
   }
   nir_deref_instr *word = cv_at(b, base, nir_iand_imm(nb, byte, ~3), glsl_uint_type());
   nir_def *shift = nir_ishl_imm(nb, nir_iand_imm(nb, byte, 2), 3);
   nir_variable *old = nir_local_variable_create(nb->impl, glsl_uint_type(), "cv_old");
   nir_store_var(nb, old, nir_load_deref_with_access(nb, word, ACCESS_VOLATILE), 1);
   nir_loop *loop = nir_push_loop(nb);
   {
      nir_def *o = nir_load_var(nb, old);
      nir_def *h = nir_u2u16(nb, nir_ushr(nb, o, shift));
      nir_def *s = nir_u2u32(nb, nir_fadd(nb, h, v));
      nir_def *mask = nir_ishl(nb, nir_imm_int(nb, 0xffff), shift);
      nir_def *n = nir_ior(nb, nir_iand(nb, o, nir_inot(nb, mask)), nir_ishl(nb, s, shift));
      nir_def *r = nir_deref_atomic_swap(nb, 32, &word->def, o, n,
                                         .atomic_op = nir_atomic_op_cmpxchg);
      nir_store_var(nb, old, r, 1);
      nir_break_if(nb, nir_ieq(nb, r, o));
   }
   nir_pop_loop(nb, loop);
}

static void
cv_outer_product(struct vtn_builder *b, const uint32_t *w, unsigned count)
{
   nir_builder *nb = &b->nb;
   nir_deref_instr *base = cv_base(b, w[1]);
   nir_def *off = nir_u2u32(nb, vtn_get_nir_ssa(b, w[2]));
   unsigned M, N;
   nir_def **a = operand(b, w[3], &M), **bv = operand(b, w[4], &N);
   const SpvCooperativeVectorMatrixLayout layout = vtn_constant_uint(b, w[5]);
   struct cv_interp t = cv_interp(b, w[6]);
   nir_def *stride = count > 7 ? vtn_get_nir_ssa(b, w[7]) : NULL;
   vtn_fail_if(t.kind != CV_FLOAT || (t.bits != 16 && t.bits != 32),
               "outer product accumulation is float16 or float32");
   const struct glsl_type *et = glsl_floatN_t_type(a[0]->bit_size);
   nir_variable *av = cv_array(b, et, a, M, "cv_a"), *bvar = cv_array(b, et, bv, N, "cv_b");
   nir_def *sm, *sn;
   cv_matrix_steps(b, layout, false, M, N, t.bits / 8, stride, &sm, &sn);

   nir_variable *iv, *jv;
   nir_loop *il, *jl;
   nir_def *i = cv_loop_begin(nb, &iv, M, &il);
   {
      nir_def *ai = nir_f2f32(nb, cv_elem(nb, av, i));
      nir_def *row = nir_iadd(nb, off, nir_imul(nb, i, sm));
      nir_def *j = cv_loop_begin(nb, &jv, N, &jl);
      {
         nir_def *p = nir_fmul(nb, ai, nir_f2f32(nb, cv_elem(nb, bvar, j)));
         cv_atomic_fadd(b, base, nir_iadd(nb, row, nir_imul(nb, j, sn)), nir_f2fN(nb, p, t.bits));
      }
      cv_loop_end(nb, jv, j, jl);
   }
   cv_loop_end(nb, iv, i, il);
}

static void
cv_reduce_sum(struct vtn_builder *b, const uint32_t *w)
{
   nir_builder *nb = &b->nb;
   nir_deref_instr *base = cv_base(b, w[1]);
   nir_def *off = nir_u2u32(nb, vtn_get_nir_ssa(b, w[2]));
   unsigned n;
   nir_def **v = operand(b, w[3], &n);
   vtn_fail_if(v[0]->bit_size != 16 && v[0]->bit_size != 32,
               "reduce sum accumulation is float16 or float32");
   nir_variable *vv = cv_array(b, glsl_floatN_t_type(v[0]->bit_size), v, n, "cv_v");
   nir_variable *iv;
   nir_loop *il;
   nir_def *i = cv_loop_begin(nb, &iv, n, &il);
   cv_atomic_fadd(b, base, nir_iadd(nb, off, nir_imul_imm(nb, i, v[0]->bit_size / 8)),
                  cv_elem(nb, vv, i));
   cv_loop_end(nb, iv, i, il);
}

void
vtn_handle_cooperative_vector_instruction(struct vtn_builder *b, SpvOp opcode,
                                          const uint32_t *w, unsigned count)
{
   nir_builder *nb = &b->nb;
   switch (opcode) {
   case SpvOpCooperativeVectorLoadNV: {
      struct vtn_type *type = vtn_get_type(b, w[1]);
      struct vtn_pointer *p = vtn_value_to_pointer(b, vtn_value(b, w[3], vtn_value_type_pointer));
      if (count > 5) {
         unsigned idx = 5, alignment;
         SpvMemoryAccessMask access = SpvMemoryAccessMaskNone;
         SpvScope scope;
         vtn_get_mem_operands(b, w, count, &idx, &access, &alignment, NULL, &scope);
         vtn_emit_make_visible_barrier(b, access, scope, p->mode);
      }
      nir_deref_instr *base = cv_base(b, w[3]);
      nir_def *off = nir_u2u32(nb, vtn_get_nir_ssa(b, w[4]));
      const struct glsl_type *et = type->array_element->type;
      const unsigned size = glsl_get_bit_size(et) / 8;
      nir_def **c = vtn_alloc_array(b, nir_def *, type->length);
      for (unsigned i = 0; i < type->length; i++)
         c[i] = nir_load_deref(nb, cv_at(b, base, nir_iadd_imm(nb, off, i * size), et));
      vtn_push_ssa_value(b, w[2], compose(b, type->type, c));
      break;
   }

   case SpvOpCooperativeVectorStoreNV: {
      struct vtn_pointer *p = vtn_value_to_pointer(b, vtn_value(b, w[1], vtn_value_type_pointer));
      nir_deref_instr *base = cv_base(b, w[1]);
      nir_def *off = nir_u2u32(nb, vtn_get_nir_ssa(b, w[2]));
      unsigned n;
      nir_def **c = operand(b, w[3], &n);
      const struct glsl_type *et = vtn_get_value_type(b, w[3])->array_element->type;
      const unsigned size = glsl_get_bit_size(et) / 8;
      for (unsigned i = 0; i < n; i++)
         nir_store_deref(nb, cv_at(b, base, nir_iadd_imm(nb, off, i * size), et), c[i], 1);
      if (count > 4) {
         unsigned idx = 4, alignment;
         SpvMemoryAccessMask access = SpvMemoryAccessMaskNone;
         SpvScope scope;
         vtn_get_mem_operands(b, w, count, &idx, &access, &alignment, &scope, NULL);
         vtn_emit_make_available_barrier(b, access, scope, p->mode);
      }
      break;
   }

   case SpvOpCooperativeVectorMatrixMulNV:
   case SpvOpCooperativeVectorMatrixMulAddNV:
      cv_matrix_mul(b, opcode, w, count);
      break;

   case SpvOpCooperativeVectorOuterProductAccumulateNV:
      cv_outer_product(b, w, count);
      break;

   case SpvOpCooperativeVectorReduceSumAccumulateNV:
      cv_reduce_sum(b, w);
      break;

   default:
      UNREACHABLE("not a cooperative vector instruction");
   }
}
