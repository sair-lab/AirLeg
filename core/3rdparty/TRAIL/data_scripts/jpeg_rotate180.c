/* Lossless 180 deg rotation of a baseline JPEG in the DCT domain (as jpegtran -rotate 180 -perfect): the block order is
 * reversed in both directions and coefficient (u, v) of every block is multiplied by (-1)^(u+v). Decoding the result
 * gives the decoded original rotated by 180 deg. Only images whose size is a multiple of the MCU (no partial edge
 * blocks) are accepted. Used by convert_grandtour_to_ros2.py (ctypes) for the upside-down GrandTour front_left images.
 *
 * build: gcc -O2 -shared -fPIC jpeg_rotate180.c -o libjpeg_rotate180.so -ljpeg
 * int jpeg_rotate180(const unsigned char* in, unsigned long in_len, unsigned char** out, unsigned long* out_len)
 *   0 on success (*out from malloc, release with jpeg_rotate180_free), 1 for a libjpeg error, 2 for a partial MCU.
 */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

struct error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf jump;
};

static void error_exit(j_common_ptr cinfo) { longjmp(((struct error_mgr*)cinfo->err)->jump, 1); }
static void silent(j_common_ptr cinfo) { (void)cinfo; }

void jpeg_rotate180_free(unsigned char* buf) { free(buf); }

int jpeg_rotate180(const unsigned char* in, unsigned long in_len, unsigned char** out, unsigned long* out_len) {
  struct jpeg_decompress_struct src;
  struct jpeg_compress_struct dst;
  struct error_mgr err;                /* shared by both objects: one setjmp */
  JBLOCK* volatile tmp = NULL;          /* modified after setjmp */
  volatile int status = 0;
  *out = NULL;
  *out_len = 0;
  src.err = dst.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = error_exit;
  err.pub.output_message = silent;
  jpeg_create_decompress(&src);
  jpeg_create_compress(&dst);
  if (setjmp(err.jump)) {
    status = 1;
    goto done;
  }
  jpeg_mem_src(&src, (unsigned char*)in, in_len);
  jpeg_read_header(&src, TRUE);
  jvirt_barray_ptr* coef = jpeg_read_coefficients(&src);
  if (src.image_width % (src.max_h_samp_factor * DCTSIZE) || src.image_height % (src.max_v_samp_factor * DCTSIZE)) {
    status = 2;
    goto done;
  }
  for (int ci = 0; ci < src.num_components; ++ci) {
    const jpeg_component_info* comp = src.comp_info + ci;
    const JDIMENSION W = comp->width_in_blocks, H = comp->height_in_blocks;
    tmp = (JBLOCK*)malloc(sizeof(JBLOCK) * W * H);
    for (JDIMENSION r = 0; r < H; ++r) {
      JBLOCKARRAY row = (*src.mem->access_virt_barray)((j_common_ptr)&src, coef[ci], r, 1, FALSE);
      memcpy(tmp + (size_t)r * W, row[0], sizeof(JBLOCK) * W);
    }
    for (JDIMENSION r = 0; r < H; ++r) {
      JBLOCKARRAY row = (*src.mem->access_virt_barray)((j_common_ptr)&src, coef[ci], r, 1, TRUE);
      for (JDIMENSION c = 0; c < W; ++c) {
        const JCOEF* from = tmp[(size_t)(H - 1 - r) * W + (W - 1 - c)];
        for (int k = 0; k < DCTSIZE2; ++k)   /* natural order: k = v * 8 + u */
          row[0][c][k] = (((k / DCTSIZE) + (k % DCTSIZE)) & 1) ? (JCOEF)-from[k] : from[k];
      }
    }
    free(tmp);
    tmp = NULL;
  }
  jpeg_copy_critical_parameters(&src, &dst);
  jpeg_mem_dest(&dst, out, out_len);
  jpeg_write_coefficients(&dst, coef);
  jpeg_finish_compress(&dst);
  jpeg_finish_decompress(&src);
done:
  free(tmp);
  if (status && *out) {
    free(*out);
    *out = NULL;
    *out_len = 0;
  }
  jpeg_destroy_compress(&dst);
  jpeg_destroy_decompress(&src);
  return status;
}
