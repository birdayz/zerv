	.text
	s_mov_b32 s11, s1
	s_movk_i32 s1, 0x8000
	s_load_b128 s[12:15], s[0:1], 0x20
	s_load_b128 s[16:19], s[0:1], 0x0
	s_load_b128 s[20:23], s[0:1], 0x30
	s_load_b128 s[24:27], s[0:1], 0x10
	s_lshl_b32 s9, s9, 8
	s_lshl_b32 s8, s8, 7
	s_lshr_b32 s28, s7, 6
	s_bfe_u32 s35, s10, 0x10014
	s_bfe_u32 s36, s10, 0x40015
	s_lshl_b32 s36, s36, 6
	s_add_u32 s36, s9, s36
	s_mul_i32 s37, s35, 0x2400
	s_lshl_b32 s35, s35, 6
	s_mov_b32 s32, 0xf000f
	s_mov_b32 s33, 0x64006400
	s_movk_i32 s34, 0xe408
	s_mov_b32 s29, 0
	s_mov_b32 s30, 0
	s_mov_b32 s31, 36
	v_mbcnt_lo_u32_b32 v160, -1, 0
	v_and_b32 v161, 15, v160
	v_mad_u32_u24 v1, 0x90, v161, s37
	v_lshrrev_b32 v162, 1, v0
	v_and_b32 v163, 1, v0
	v_add_nc_u32 v164, s8, v162
	v_mul_lo_u32 v164, v164, s2
	v_add_nc_u32 v164, s11, v164
	v_mad_u32_u24 v164, 18, v163, v164
	v_and_b32 v3, -4, v164
	v_and_b32 v165, 2, v164
	v_lshlrev_b32 v10, 3, v165
	v_mul_u32_u24 v165, 0x10001, v165
	v_add_nc_u32 v8, 0xc030c02, v165
	v_add_nc_u32 v9, 0xc050c04, v165
	v_mul_u32_u24 v2, 0x90, v162
	v_lshl_add_u32 v2, v163, 6, v2
	s_waitcnt lgkmcnt(0)
	s_bitset0_b32 s13, 14
	s_buffer_load_b32 s12, s[12:15], 0x8
	s_lshr_b32 s38, s22, 4
	s_lshr_b32 s39, s7, 3
	s_sub_i32 s38, s38, s39
	v_add3_u32 v4, s36, 0, v161
	v_mul_lo_u32 v4, v4, s4
	v_add_nc_u32 v4, s3, v4
	v_lshrrev_b32 v4, 3, v4
	v_min_u32 v4, s38, v4
	v_lshlrev_b32 v4, 4, v4
	v_add3_u32 v5, s36, 16, v161
	v_mul_lo_u32 v5, v5, s4
	v_add_nc_u32 v5, s3, v5
	v_lshrrev_b32 v5, 3, v5
	v_min_u32 v5, s38, v5
	v_lshlrev_b32 v5, 4, v5
	v_add3_u32 v6, s36, 32, v161
	v_mul_lo_u32 v6, v6, s4
	v_add_nc_u32 v6, s3, v6
	v_lshrrev_b32 v6, 3, v6
	v_min_u32 v6, s38, v6
	v_lshlrev_b32 v6, 4, v6
	v_add3_u32 v7, s36, 48, v161
	v_mul_lo_u32 v7, v7, s4
	v_add_nc_u32 v7, s3, v7
	v_lshrrev_b32 v7, 3, v7
	v_min_u32 v7, s38, v7
	v_lshlrev_b32 v7, 4, v7
	buffer_load_b128 v[11:14], v3, s[16:19], 0 offen
	buffer_load_b32 v15, v3, s[16:19], 0 offen offset:16
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:0
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:16
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:0
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:16
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:0
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:16
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:0
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:16
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:32
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:48
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:32
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:48
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:32
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:48
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:32
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:48
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v75, 0
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v77, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v81, 0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v83, 0
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v85, 0
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v87, 0
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v89, 0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v93, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v97, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v159, 0
	s_waitcnt vmcnt(17)
	v_perm_b32 v16, v12, v11, v8
	v_perm_b32 v17, v12, v11, v9
	v_perm_b32 v18, v13, v12, v8
	v_perm_b32 v19, v13, v12, v9
	v_perm_b32 v20, v14, v13, v8
	v_perm_b32 v21, v14, v13, v9
	s_waitcnt vmcnt(16)
	v_perm_b32 v22, v15, v14, v8
	v_perm_b32 v23, v15, v14, v9
	v_lshrrev_b32 v0, v10, v11
	v_lshrrev_b32 v24, 4, v16
	v_lshrrev_b32 v25, 4, v17
	v_lshrrev_b32 v26, 4, v18
	v_lshrrev_b32 v27, 4, v19
	v_lshrrev_b32 v28, 4, v20
	v_lshrrev_b32 v29, 4, v21
	v_lshrrev_b32 v30, 4, v22
	v_lshrrev_b32 v31, 4, v23
	v_and_or_b32 v16, v16, s32, s33
	v_and_or_b32 v17, v17, s32, s33
	v_and_or_b32 v18, v18, s32, s33
	v_and_or_b32 v19, v19, s32, s33
	v_and_or_b32 v20, v20, s32, s33
	v_and_or_b32 v21, v21, s32, s33
	v_and_or_b32 v22, v22, s32, s33
	v_and_or_b32 v23, v23, s32, s33
	v_and_or_b32 v24, v24, s32, s33
	v_and_or_b32 v25, v25, s32, s33
	v_and_or_b32 v26, v26, s32, s33
	v_and_or_b32 v27, v27, s32, s33
	v_and_or_b32 v28, v28, s32, s33
	v_and_or_b32 v29, v29, s32, s33
	v_and_or_b32 v30, v30, s32, s33
	v_and_or_b32 v31, v31, s32, s33
	v_pk_add_f16 v16, v16, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v17, v17, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v18, v18, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v19, v19, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v20, v20, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v21, v21, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v22, v22, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v23, v23, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v24, v24, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v25, v25, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v26, v26, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v27, v27, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v28, v28, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v29, v29, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v30, v30, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v31, v31, s34 op_sel_hi:[1,0]
	v_pk_mul_f16 v16, v16, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v17, v17, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v18, v18, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v19, v19, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v20, v20, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v21, v21, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v22, v22, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v23, v23, v0 op_sel_hi:[1,0]
	ds_store_b128 v2, v[16:19] offset:0
	ds_store_b128 v2, v[20:23] offset:16
	v_pk_mul_f16 v24, v24, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v25, v25, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v26, v26, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v27, v27, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v28, v28, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v29, v29, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v30, v30, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v31, v31, v0 op_sel_hi:[1,0]
	ds_store_b128 v2, v[24:27] offset:32
	ds_store_b128 v2, v[28:31] offset:48
	s_waitcnt_depctr 0xffe3
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_u32 s9, s12
	s_cbranch_scc0 .Lend
	s_add_u32 s47, s12, -1
	s_add_u32 s40, s9, 0x100
	s_cmp_le_u32 s40, s12
	s_cbranch_scc1 .Lxok
	s_waitcnt vmcnt(0)
	v_mbcnt_lo_u32_b32 v16, -1, 0
	v_and_b32 v17, 15, v16
	v_add3_u32 v4, s36, 0, v17
	v_min_u32 v4, s47, v4
	v_mul_lo_u32 v4, v4, s4
	v_add_nc_u32 v4, s3, v4
	v_lshrrev_b32 v4, 3, v4
	v_min_u32 v4, s38, v4
	v_lshlrev_b32 v4, 4, v4
	v_add3_u32 v5, s36, 16, v17
	v_min_u32 v5, s47, v5
	v_mul_lo_u32 v5, v5, s4
	v_add_nc_u32 v5, s3, v5
	v_lshrrev_b32 v5, 3, v5
	v_min_u32 v5, s38, v5
	v_lshlrev_b32 v5, 4, v5
	v_add3_u32 v6, s36, 32, v17
	v_min_u32 v6, s47, v6
	v_mul_lo_u32 v6, v6, s4
	v_add_nc_u32 v6, s3, v6
	v_lshrrev_b32 v6, 3, v6
	v_min_u32 v6, s38, v6
	v_lshlrev_b32 v6, 4, v6
	v_add3_u32 v7, s36, 48, v17
	v_min_u32 v7, s47, v7
	v_mul_lo_u32 v7, v7, s4
	v_add_nc_u32 v7, s3, v7
	v_lshrrev_b32 v7, 3, v7
	v_min_u32 v7, s38, v7
	v_lshlrev_b32 v7, 4, v7
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:0
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:16
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:0
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:16
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:0
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:16
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:0
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:16
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:32
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:48
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:32
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:48
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:32
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:48
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:32
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:48
.Lxok:
	ds_load_b128 v[224:227], v1 offset:0
	ds_load_b128 v[228:231], v1 offset:16
	ds_load_b128 v[232:235], v1 offset:2304
	ds_load_b128 v[236:239], v1 offset:2320
	ds_load_b128 v[240:243], v1 offset:4608
	ds_load_b128 v[244:247], v1 offset:4624
	ds_load_b128 v[248:251], v1 offset:6912
	ds_load_b128 v[252:255], v1 offset:6928
	s_cmp_eq_u32 s28, 1
	s_cbranch_scc1 .Llast0
.Lloop:
	buffer_load_b128 v[11:14], v3, s[16:19], s31 offen
	buffer_load_b32 v15, v3, s[16:19], s31 offen offset:16
	s_setprio 1
	s_waitcnt vmcnt(16) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(14)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:32
	ds_load_b128 v[228:231], v1 offset:48
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:2336
	ds_load_b128 v[236:239], v1 offset:2352
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:4640
	ds_load_b128 v[244:247], v1 offset:4656
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:64
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:80
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:64
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:80
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:64
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:80
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:64
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:80
	ds_load_b128 v[248:251], v1 offset:6944
	ds_load_b128 v[252:255], v1 offset:6960
	s_setprio 1
	s_waitcnt vmcnt(16) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(14)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:64
	ds_load_b128 v[228:231], v1 offset:80
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:2368
	ds_load_b128 v[236:239], v1 offset:2384
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:4672
	ds_load_b128 v[244:247], v1 offset:4688
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:96
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:112
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:96
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:112
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:96
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:112
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:96
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:112
	ds_load_b128 v[248:251], v1 offset:6976
	ds_load_b128 v[252:255], v1 offset:6992
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	v_perm_b32 v16, v12, v11, v8
	v_perm_b32 v17, v12, v11, v9
	v_perm_b32 v18, v13, v12, v8
	v_perm_b32 v19, v13, v12, v9
	v_perm_b32 v20, v14, v13, v8
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	v_perm_b32 v21, v14, v13, v9
	v_perm_b32 v22, v15, v14, v8
	v_perm_b32 v23, v15, v14, v9
	v_lshrrev_b32 v0, v10, v11
	v_lshrrev_b32 v24, 4, v16
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	v_lshrrev_b32 v25, 4, v17
	v_lshrrev_b32 v26, 4, v18
	v_lshrrev_b32 v27, 4, v19
	v_lshrrev_b32 v28, 4, v20
	v_lshrrev_b32 v29, 4, v21
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	v_lshrrev_b32 v30, 4, v22
	v_lshrrev_b32 v31, 4, v23
	v_and_or_b32 v16, v16, s32, s33
	v_and_or_b32 v17, v17, s32, s33
	v_and_or_b32 v18, v18, s32, s33
	ds_load_b128 v[224:227], v1 offset:96
	ds_load_b128 v[228:231], v1 offset:112
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	v_and_or_b32 v19, v19, s32, s33
	v_and_or_b32 v20, v20, s32, s33
	v_and_or_b32 v21, v21, s32, s33
	v_and_or_b32 v22, v22, s32, s33
	v_and_or_b32 v23, v23, s32, s33
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	v_and_or_b32 v24, v24, s32, s33
	v_and_or_b32 v25, v25, s32, s33
	v_and_or_b32 v26, v26, s32, s33
	v_and_or_b32 v27, v27, s32, s33
	v_and_or_b32 v28, v28, s32, s33
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	v_and_or_b32 v29, v29, s32, s33
	v_and_or_b32 v30, v30, s32, s33
	v_and_or_b32 v31, v31, s32, s33
	v_pk_add_f16 v16, v16, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v17, v17, s34 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	v_pk_add_f16 v18, v18, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v19, v19, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v20, v20, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v21, v21, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v22, v22, s34 op_sel_hi:[1,0]
	ds_load_b128 v[232:235], v1 offset:2400
	ds_load_b128 v[236:239], v1 offset:2416
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	v_pk_add_f16 v23, v23, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v24, v24, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v25, v25, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v26, v26, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v27, v27, s34 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	v_pk_add_f16 v28, v28, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v29, v29, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v30, v30, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v31, v31, s34 op_sel_hi:[1,0]
	v_pk_mul_f16 v16, v16, v0 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	v_pk_mul_f16 v17, v17, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v18, v18, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v19, v19, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v20, v20, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v21, v21, v0 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	v_pk_mul_f16 v22, v22, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v23, v23, v0 op_sel_hi:[1,0]
	ds_store_b128 v2, v[16:19] offset:18432
	ds_store_b128 v2, v[20:23] offset:18448
	v_pk_mul_f16 v24, v24, v0 op_sel_hi:[1,0]
	ds_load_b128 v[240:243], v1 offset:4704
	ds_load_b128 v[244:247], v1 offset:4720
	s_setprio 1
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	v_pk_mul_f16 v25, v25, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v26, v26, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v27, v27, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v28, v28, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v29, v29, v0 op_sel_hi:[1,0]
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:128
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:144
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	v_pk_mul_f16 v30, v30, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v31, v31, v0 op_sel_hi:[1,0]
	ds_store_b128 v2, v[24:27] offset:18464
	ds_store_b128 v2, v[28:31] offset:18480
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:128
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:144
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:128
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:144
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:128
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:144
	ds_load_b128 v[248:251], v1 offset:7008
	ds_load_b128 v[252:255], v1 offset:7024
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	s_waitcnt_depctr 0xffe3
	s_barrier
	ds_load_b128 v[224:227], v1 offset:18432
	ds_load_b128 v[228:231], v1 offset:18448
	s_setprio 1
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:20736
	ds_load_b128 v[236:239], v1 offset:20752
	s_setprio 1
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:23040
	ds_load_b128 v[244:247], v1 offset:23056
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:160
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:176
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:160
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:176
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:160
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:176
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:160
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:176
	ds_load_b128 v[248:251], v1 offset:25344
	ds_load_b128 v[252:255], v1 offset:25360
	s_add_u32 s41, s29, 2
	s_cmp_eq_u32 s41, s28
	s_cbranch_scc1 .Llast1
	buffer_load_b128 v[11:14], v3, s[16:19], s31 offen offset:36
	buffer_load_b32 v15, v3, s[16:19], s31 offen offset:52
	s_setprio 1
	s_waitcnt vmcnt(16) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(14)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:18464
	ds_load_b128 v[228:231], v1 offset:18480
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:20768
	ds_load_b128 v[236:239], v1 offset:20784
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:23072
	ds_load_b128 v[244:247], v1 offset:23088
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:192
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:208
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:192
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:208
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:192
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:208
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:192
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:208
	ds_load_b128 v[248:251], v1 offset:25376
	ds_load_b128 v[252:255], v1 offset:25392
	s_setprio 1
	s_waitcnt vmcnt(16) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(14)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:18496
	ds_load_b128 v[228:231], v1 offset:18512
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:20800
	ds_load_b128 v[236:239], v1 offset:20816
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:23104
	ds_load_b128 v[244:247], v1 offset:23120
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:224
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:240
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:224
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:240
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:224
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:240
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:224
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:240
	ds_load_b128 v[248:251], v1 offset:25408
	ds_load_b128 v[252:255], v1 offset:25424
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	v_perm_b32 v16, v12, v11, v8
	v_perm_b32 v17, v12, v11, v9
	v_perm_b32 v18, v13, v12, v8
	v_perm_b32 v19, v13, v12, v9
	v_perm_b32 v20, v14, v13, v8
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	v_perm_b32 v21, v14, v13, v9
	v_perm_b32 v22, v15, v14, v8
	v_perm_b32 v23, v15, v14, v9
	v_lshrrev_b32 v0, v10, v11
	v_lshrrev_b32 v24, 4, v16
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	v_lshrrev_b32 v25, 4, v17
	v_lshrrev_b32 v26, 4, v18
	v_lshrrev_b32 v27, 4, v19
	v_lshrrev_b32 v28, 4, v20
	v_lshrrev_b32 v29, 4, v21
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	v_lshrrev_b32 v30, 4, v22
	v_lshrrev_b32 v31, 4, v23
	v_and_or_b32 v16, v16, s32, s33
	v_and_or_b32 v17, v17, s32, s33
	v_and_or_b32 v18, v18, s32, s33
	ds_load_b128 v[224:227], v1 offset:18528
	ds_load_b128 v[228:231], v1 offset:18544
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	v_and_or_b32 v19, v19, s32, s33
	v_and_or_b32 v20, v20, s32, s33
	v_and_or_b32 v21, v21, s32, s33
	v_and_or_b32 v22, v22, s32, s33
	v_and_or_b32 v23, v23, s32, s33
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	v_and_or_b32 v24, v24, s32, s33
	v_and_or_b32 v25, v25, s32, s33
	v_and_or_b32 v26, v26, s32, s33
	v_and_or_b32 v27, v27, s32, s33
	v_and_or_b32 v28, v28, s32, s33
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	v_and_or_b32 v29, v29, s32, s33
	v_and_or_b32 v30, v30, s32, s33
	v_and_or_b32 v31, v31, s32, s33
	v_pk_add_f16 v16, v16, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v17, v17, s34 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	v_pk_add_f16 v18, v18, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v19, v19, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v20, v20, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v21, v21, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v22, v22, s34 op_sel_hi:[1,0]
	ds_load_b128 v[232:235], v1 offset:20832
	ds_load_b128 v[236:239], v1 offset:20848
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	v_pk_add_f16 v23, v23, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v24, v24, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v25, v25, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v26, v26, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v27, v27, s34 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	v_pk_add_f16 v28, v28, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v29, v29, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v30, v30, s34 op_sel_hi:[1,0]
	v_pk_add_f16 v31, v31, s34 op_sel_hi:[1,0]
	v_pk_mul_f16 v16, v16, v0 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	v_pk_mul_f16 v17, v17, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v18, v18, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v19, v19, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v20, v20, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v21, v21, v0 op_sel_hi:[1,0]
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	v_pk_mul_f16 v22, v22, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v23, v23, v0 op_sel_hi:[1,0]
	ds_store_b128 v2, v[16:19] offset:0
	ds_store_b128 v2, v[20:23] offset:16
	v_pk_mul_f16 v24, v24, v0 op_sel_hi:[1,0]
	ds_load_b128 v[240:243], v1 offset:23136
	ds_load_b128 v[244:247], v1 offset:23152
	s_setprio 1
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	v_pk_mul_f16 v25, v25, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v26, v26, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v27, v27, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v28, v28, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v29, v29, v0 op_sel_hi:[1,0]
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:256
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:272
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	v_pk_mul_f16 v30, v30, v0 op_sel_hi:[1,0]
	v_pk_mul_f16 v31, v31, v0 op_sel_hi:[1,0]
	ds_store_b128 v2, v[24:27] offset:32
	ds_store_b128 v2, v[28:31] offset:48
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:256
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:272
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:256
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:272
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:256
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:272
	ds_load_b128 v[248:251], v1 offset:25440
	ds_load_b128 v[252:255], v1 offset:25456
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	s_waitcnt_depctr 0xffe3
	s_barrier
	ds_load_b128 v[224:227], v1 offset:0
	ds_load_b128 v[228:231], v1 offset:16
	s_setprio 1
	s_waitcnt lgkmcnt(10)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:2304
	ds_load_b128 v[236:239], v1 offset:2320
	s_setprio 1
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:4608
	ds_load_b128 v[244:247], v1 offset:4624
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:288
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:304
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:288
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:304
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:288
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:304
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:288
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:304
	ds_load_b128 v[248:251], v1 offset:6912
	ds_load_b128 v[252:255], v1 offset:6928
	s_add_u32 s29, s29, 2
	s_addk_i32 s30, 0x100
	s_add_u32 s31, s31, 72
	s_add_u32 s41, s29, 1
	s_cmp_lg_u32 s41, s28
	s_cbranch_scc1 .Lloop
.Llast0:
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:32
	ds_load_b128 v[228:231], v1 offset:48
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:2336
	ds_load_b128 v[236:239], v1 offset:2352
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:4640
	ds_load_b128 v[244:247], v1 offset:4656
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:64
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:80
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:64
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:80
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:64
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:80
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:64
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:80
	ds_load_b128 v[248:251], v1 offset:6944
	ds_load_b128 v[252:255], v1 offset:6960
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:64
	ds_load_b128 v[228:231], v1 offset:80
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:2368
	ds_load_b128 v[236:239], v1 offset:2384
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:4672
	ds_load_b128 v[244:247], v1 offset:4688
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:96
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:112
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:96
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:112
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:96
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:112
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:96
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:112
	ds_load_b128 v[248:251], v1 offset:6976
	ds_load_b128 v[252:255], v1 offset:6992
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:96
	ds_load_b128 v[228:231], v1 offset:112
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:2400
	ds_load_b128 v[236:239], v1 offset:2416
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:4704
	ds_load_b128 v[244:247], v1 offset:4720
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	ds_load_b128 v[248:251], v1 offset:7008
	ds_load_b128 v[252:255], v1 offset:7024
	s_setprio 1
	s_waitcnt vmcnt(6) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(4)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	s_setprio 1
	s_waitcnt lgkmcnt(2)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	s_movk_i32 s42, 18432
	s_branch .Lepi
.Llast1:
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:18464
	ds_load_b128 v[228:231], v1 offset:18480
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:20768
	ds_load_b128 v[236:239], v1 offset:20784
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:23072
	ds_load_b128 v[244:247], v1 offset:23088
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	buffer_load_b128 v[160:163], v4, s[20:23], s30 offen offset:192
	buffer_load_b128 v[164:167], v4, s[20:23], s30 offen offset:208
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	buffer_load_b128 v[168:171], v5, s[20:23], s30 offen offset:192
	buffer_load_b128 v[172:175], v5, s[20:23], s30 offen offset:208
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	buffer_load_b128 v[176:179], v6, s[20:23], s30 offen offset:192
	buffer_load_b128 v[180:183], v6, s[20:23], s30 offen offset:208
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	buffer_load_b128 v[184:187], v7, s[20:23], s30 offen offset:192
	buffer_load_b128 v[188:191], v7, s[20:23], s30 offen offset:208
	ds_load_b128 v[248:251], v1 offset:25376
	ds_load_b128 v[252:255], v1 offset:25392
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:18496
	ds_load_b128 v[228:231], v1 offset:18512
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:20800
	ds_load_b128 v[236:239], v1 offset:20816
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:23104
	ds_load_b128 v[244:247], v1 offset:23120
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	buffer_load_b128 v[192:195], v4, s[20:23], s30 offen offset:224
	buffer_load_b128 v[196:199], v4, s[20:23], s30 offen offset:240
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	buffer_load_b128 v[200:203], v5, s[20:23], s30 offen offset:224
	buffer_load_b128 v[204:207], v5, s[20:23], s30 offen offset:240
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	buffer_load_b128 v[208:211], v6, s[20:23], s30 offen offset:224
	buffer_load_b128 v[212:215], v6, s[20:23], s30 offen offset:240
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	buffer_load_b128 v[216:219], v7, s[20:23], s30 offen offset:224
	buffer_load_b128 v[220:223], v7, s[20:23], s30 offen offset:240
	ds_load_b128 v[248:251], v1 offset:25408
	ds_load_b128 v[252:255], v1 offset:25424
	s_setprio 1
	s_waitcnt vmcnt(14) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[160:167], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(12)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[168:175], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[176:183], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(8)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[184:191], v[56:63]
	s_setprio 0
	ds_load_b128 v[224:227], v1 offset:18528
	ds_load_b128 v[228:231], v1 offset:18544
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[160:167], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[168:175], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[176:183], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[184:191], v[88:95]
	s_setprio 0
	ds_load_b128 v[232:235], v1 offset:20832
	ds_load_b128 v[236:239], v1 offset:20848
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[160:167], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[168:175], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[176:183], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[184:191], v[120:127]
	s_setprio 0
	ds_load_b128 v[240:243], v1 offset:23136
	ds_load_b128 v[244:247], v1 offset:23152
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[160:167], v[128:135]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[168:175], v[136:143]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[176:183], v[144:151]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[184:191], v[152:159]
	s_setprio 0
	ds_load_b128 v[248:251], v1 offset:25440
	ds_load_b128 v[252:255], v1 offset:25456
	s_setprio 1
	s_waitcnt vmcnt(6) lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[32:39], v[224:231], v[192:199], v[32:39]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(4)
	v_wmma_f32_16x16x16_f16 v[40:47], v[224:231], v[200:207], v[40:47]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_wmma_f32_16x16x16_f16 v[48:55], v[224:231], v[208:215], v[48:55]
	s_setprio 0
	s_setprio 1
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[56:63], v[224:231], v[216:223], v[56:63]
	s_setprio 0
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_wmma_f32_16x16x16_f16 v[64:71], v[232:239], v[192:199], v[64:71]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[72:79], v[232:239], v[200:207], v[72:79]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[80:87], v[232:239], v[208:215], v[80:87]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[88:95], v[232:239], v[216:223], v[88:95]
	s_setprio 0
	s_setprio 1
	s_waitcnt lgkmcnt(2)
	v_wmma_f32_16x16x16_f16 v[96:103], v[240:247], v[192:199], v[96:103]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[200:207], v[104:111]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[208:215], v[112:119]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127]
	s_setprio 0
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[192:199], v[128:135]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[200:207], v[136:143]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[208:215], v[144:151]
	s_setprio 0
	s_setprio 1
	v_wmma_f32_16x16x16_f16 v[152:159], v[248:255], v[216:223], v[152:159]
	s_setprio 0
	s_mov_b32 s42, 0
.Lepi:
	s_bfe_u32 s43, s10, 0x50014
	s_mulk_i32 s43, 0x900
	s_add_u32 s42, s42, s43
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_and_b32 v1, 15, v0
	v_lshrrev_b32 v2, 4, v0
	v_mad_u32_u24 v3, 0x90, v1, s42
	v_lshl_add_u32 v3, v2, 2, v3
	v_lshrrev_b32 v4, 3, v0
	v_and_b32 v5, 7, v0
	v_mad_u32_u24 v6, 0x90, v4, s42
	v_lshl_add_u32 v6, v5, 4, v6
	v_mul_lo_u32 v7, v4, s6
	v_lshl_add_u32 v7, v5, 2, v7
	v_lshlrev_b32 v7, 2, v7
	s_add_u32 s44, s5, s8
	s_add_u32 s44, s44, s35
	ds_store_2addr_b32 v3, v32, v33 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v34, v35 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v36, v37 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v38, v39 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v64, v65 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v66, v67 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v68, v69 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v70, v71 offset0:28 offset1:30
	ds_load_b128 v[160:163], v6 offset:0
	ds_load_b128 v[164:167], v6 offset:576
	ds_load_b128 v[168:171], v6 offset:1152
	ds_load_b128 v[172:175], v6 offset:1728
	s_add_u32 s45, s36, 0
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[160:163], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 4
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[164:167], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 8
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[168:171], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 12
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[172:175], v7, s[24:27], s45 offen
	ds_store_2addr_b32 v3, v96, v97 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v98, v99 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v100, v101 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v102, v103 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v128, v129 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v130, v131 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v132, v133 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v134, v135 offset0:28 offset1:30
	ds_load_b128 v[176:179], v6 offset:0
	ds_load_b128 v[180:183], v6 offset:576
	ds_load_b128 v[184:187], v6 offset:1152
	ds_load_b128 v[188:191], v6 offset:1728
	s_add_u32 s45, s36, 0
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[176:179], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 4
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[180:183], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 8
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[184:187], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 12
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[188:191], v7, s[24:27], s45 offen offset:128
	ds_store_2addr_b32 v3, v40, v41 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v42, v43 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v44, v45 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v46, v47 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v72, v73 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v74, v75 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v76, v77 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v78, v79 offset0:28 offset1:30
	ds_load_b128 v[192:195], v6 offset:0
	ds_load_b128 v[196:199], v6 offset:576
	ds_load_b128 v[200:203], v6 offset:1152
	ds_load_b128 v[204:207], v6 offset:1728
	s_add_u32 s45, s36, 16
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[192:195], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 20
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[196:199], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 24
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[200:203], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 28
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[204:207], v7, s[24:27], s45 offen
	ds_store_2addr_b32 v3, v104, v105 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v106, v107 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v108, v109 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v110, v111 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v136, v137 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v138, v139 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v140, v141 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v142, v143 offset0:28 offset1:30
	ds_load_b128 v[208:211], v6 offset:0
	ds_load_b128 v[212:215], v6 offset:576
	ds_load_b128 v[216:219], v6 offset:1152
	ds_load_b128 v[220:223], v6 offset:1728
	s_add_u32 s45, s36, 16
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[208:211], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 20
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[212:215], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 24
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[216:219], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 28
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[220:223], v7, s[24:27], s45 offen offset:128
	ds_store_2addr_b32 v3, v48, v49 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v50, v51 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v52, v53 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v54, v55 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v80, v81 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v82, v83 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v84, v85 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v86, v87 offset0:28 offset1:30
	ds_load_b128 v[224:227], v6 offset:0
	ds_load_b128 v[228:231], v6 offset:576
	ds_load_b128 v[232:235], v6 offset:1152
	ds_load_b128 v[236:239], v6 offset:1728
	s_add_u32 s45, s36, 32
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[224:227], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 36
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[228:231], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 40
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[232:235], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 44
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[236:239], v7, s[24:27], s45 offen
	ds_store_2addr_b32 v3, v112, v113 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v114, v115 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v116, v117 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v118, v119 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v144, v145 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v146, v147 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v148, v149 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v150, v151 offset0:28 offset1:30
	ds_load_b128 v[240:243], v6 offset:0
	ds_load_b128 v[244:247], v6 offset:576
	ds_load_b128 v[248:251], v6 offset:1152
	ds_load_b128 v[252:255], v6 offset:1728
	s_add_u32 s45, s36, 32
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[240:243], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 36
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[244:247], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 40
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[248:251], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 44
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[252:255], v7, s[24:27], s45 offen offset:128
	ds_store_2addr_b32 v3, v56, v57 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v58, v59 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v60, v61 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v62, v63 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v88, v89 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v90, v91 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v92, v93 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v94, v95 offset0:28 offset1:30
	ds_load_b128 v[8:11], v6 offset:0
	ds_load_b128 v[12:15], v6 offset:576
	ds_load_b128 v[16:19], v6 offset:1152
	ds_load_b128 v[20:23], v6 offset:1728
	s_add_u32 s45, s36, 48
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[8:11], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 52
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[12:15], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 56
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[16:19], v7, s[24:27], s45 offen
	s_add_u32 s45, s36, 60
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[20:23], v7, s[24:27], s45 offen
	ds_store_2addr_b32 v3, v120, v121 offset0:0 offset1:2
	ds_store_2addr_b32 v3, v122, v123 offset0:4 offset1:6
	ds_store_2addr_b32 v3, v124, v125 offset0:8 offset1:10
	ds_store_2addr_b32 v3, v126, v127 offset0:12 offset1:14
	ds_store_2addr_b32 v3, v152, v153 offset0:16 offset1:18
	ds_store_2addr_b32 v3, v154, v155 offset0:20 offset1:22
	ds_store_2addr_b32 v3, v156, v157 offset0:24 offset1:26
	ds_store_2addr_b32 v3, v158, v159 offset0:28 offset1:30
	s_waitcnt_depctr 0xffe3
	ds_load_b128 v[160:163], v6 offset:0
	ds_load_b128 v[164:167], v6 offset:576
	ds_load_b128 v[168:171], v6 offset:1152
	ds_load_b128 v[172:175], v6 offset:1728
	s_add_u32 s45, s36, 48
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(3)
	buffer_store_b128 v[160:163], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 52
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(2)
	buffer_store_b128 v[164:167], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 56
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(1)
	buffer_store_b128 v[168:171], v7, s[24:27], s45 offen offset:128
	s_add_u32 s45, s36, 60
	s_mul_i32 s45, s45, s6
	s_add_u32 s45, s45, s44
	s_lshl_b32 s45, s45, 2
	s_waitcnt lgkmcnt(0)
	buffer_store_b128 v[172:175], v7, s[24:27], s45 offen offset:128
.Lend:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
