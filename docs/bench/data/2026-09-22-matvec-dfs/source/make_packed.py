from pathlib import Path
base=Path(__file__).with_name('block.comp').read_text()
s=base[:base.index('void main()')]+'''uint word_at(uint b) {
 uint lo=weights[b>>2];
 if((b&2)==0) return lo;
 return (lo>>16)|(weights[(b>>2)+1]<<16);
}
vec4 accum4(vec4 s, vec4 w, vec4 x) {
#if ACC_FMA
 return fma(w,x,s);
#else
 precise vec4 product=w*x;
 precise vec4 added=s+product;
 return added;
#endif
}
vec4 input4(uint c) { return vec4(input_values[c],input_values[c+1],input_values[c+2],input_values[c+3]); }
vec4 coeff4(uint w,uint mask,uint bias) {
 return vec4(ivec4(uvec4(w,w>>8,w>>16,w>>24)&mask)-int(bias));
}
void main() {
 uint tid=gl_LocalInvocationID.x, lane=tid%LANES;
 uint row=(gl_WorkGroupID.y*p.groups_x+gl_WorkGroupID.x)*ROWS+tid/LANES;
 vec4 a=vec4(0),z=vec4(0);
 if(row<p.rows) {
 uint row_start=p.weight_offset+row*p.row_bytes;
#if FORMAT==0
 for(uint c=lane*4;c<p.columns;c+=LANES*4) {
  for(uint k=0;k<4 && c+k<p.columns;k++) a[k]=accum(a[k],uintBitsToFloat(weights[(row_start>>2)+c+k]),input_values[p.input_offset+c+k]);
 }
#elif FORMAT==2 || FORMAT==3
 for(uint block=lane/4;block<p.columns/32;block+=LANES/4) {
  uint b=row_start+block*BLOCK_BYTES,j=(lane%4)*4;
  uint q=word_at(b+PAYLOAD_OFFSET+j);
  float d=half_at(b);
#if FORMAT==2
  precise vec4 w0=d*coeff4(q,15,8),w1=d*coeff4(q>>4,15,8);
#else
  float minimum=half_at(b+2);
  precise vec4 prod0=d*coeff4(q,15,0),prod1=d*coeff4(q>>4,15,0);
  precise vec4 w0=prod0+minimum,w1=prod1+minimum;
#endif
  a=accum4(a,w0,input4(p.input_offset+block*32+j));
  z=accum4(z,w1,input4(p.input_offset+block*32+j+16));
 }
#elif FORMAT==13 || FORMAT==14
 for(uint block=lane/8;block<p.columns/256;block+=LANES/8) {
  uint b=row_start+block*BLOCK_BYTES,j=(lane%8)*4;
#if FORMAT==13
  float d=half_at(b),minimum=half_at(b+2);
  uint high=word_at(b+16+j);
'''
for g in range(8):
 s+=f'  uint q{g}=((word_at(b+{48+(g//2)*32}+j)>>{4*(g%2)})&0x0f0f0f0f)|(((high>>{g})&0x01010101)<<4);\n'
 if g<4:s+=f'  uint sc{g}=byte_at(b+{4+g})&63,mn{g}=byte_at(b+{8+g})&63;\n'
 else:s+=f'  uint sc{g}=(byte_at(b+{8+g})&15)|((byte_at(b+{g})>>6)<<4),mn{g}=(byte_at(b+{8+g})>>4)|((byte_at(b+{4+g})>>6)<<4);\n'
 s+=f'  precise float scale{g}=d*float(sc{g}),bias{g}=minimum*float(mn{g});\n  precise vec4 prod{g}=scale{g}*coeff4(q{g},31,0),w{g}=prod{g}-bias{g};\n  {"a" if g%2==0 else "z"}=accum4({"a" if g%2==0 else "z"},w{g},input4(p.input_offset+block*256+{g*32}+j));\n'
s+='''#else
  float d=half_at(b+208);
  uint lo0=word_at(b+j),lo1=word_at(b+32+j),lo2=word_at(b+64+j),lo3=word_at(b+96+j);
  uint hi0=word_at(b+128+j),hi1=word_at(b+160+j);
'''
for g in range(8):
 lo=g//4*2+g%2;hi=g//4;shift=(g%4)*2
 s+=f'  uint q{g}=((lo{lo}>>{4*((g%4)//2)})&0x0f0f0f0f)|(((hi{hi}>>{shift})&0x03030303)<<4);\n  precise float scale{g}=d*float(signed_byte(byte_at(b+{192+g*2}+j/16)));\n  precise vec4 w{g}=scale{g}*coeff4(q{g},63,32);\n  {"a" if g%2==0 else "z"}=accum4({"a" if g%2==0 else "z"},w{g},input4(p.input_offset+block*256+{g*32}+j));\n'
s+='''#endif
 }
#endif
 }
 precise vec4 v=a+z;
 precise float sum=(v.x+v.y)+(v.z+v.w);
 partials[tid]=sum;
 barrier();
 for(uint stride=LANES/2;stride>0;stride>>=1) {
  if(lane<stride) { precise float value=partials[tid]+partials[tid+stride]; partials[tid]=value; }
  barrier();
 }
 if(lane==0 && row<p.rows) output_values[p.output_offset+row]=partials[tid];
}
'''
Path(__file__).with_name('packed.comp').write_text(s)
