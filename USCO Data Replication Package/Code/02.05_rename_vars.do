********************************************************************************
* Rename datafields                                                            *
* Last modified April 10, 2024                                                 *
********************************************************************************

capture confirm variable v_df_257_sf_a_rep_1
if _rc != 0 {
	gen v_df_257_sf_a_rep_1 = ""
}
rename v_df_257_sf_a_rep_1 place_of_1st_pub
capture drop v_df_257_sf_a_rep*

capture confirm variable v_df_549_sf_a_rep_1
if _rc != 0 {
	gen v_df_549_sf_a_rep_1 = ""
}
rename v_df_549_sf_a_rep_1 scope_of_rights
capture drop v_df_549_sf_a_rep_*

capture confirm variable v_df_589_sf_a_rep_1
if _rc != 0 {
	gen v_df_589_sf_a_rep_1 = ""
}
rename v_df_589_sf_a_rep_1 permissions_contact
capture drop v_df_589_sf_a_rep_*

capture confirm variable v_df_917_sf_b_rep_1
if _rc != 0 {
	gen v_df_917_sf_b_rep_1 = ""
}
rename v_df_917_sf_b_rep_1 work_type
capture drop v_df_917_sf_b_rep_*

capture confirm variable v_df_917_sf_p_rep_1
if _rc != 0 {
	gen v_df_917_sf_p_rep_1 = ""
}
rename v_df_917_sf_p_rep_1 publication_stat
capture drop v_df_917_sf_p_rep_*

capture confirm variable v_num_authors
if _rc != 0 {
	gen v_num_authors = ""
}

rename v_num_authors num_authors
rename v_num_claimants num_claimants
rename v_title title
rename v_reg_num reg_num

gen reg_date = ""
capture replace reg_date = v_df_017_sf_d_rep_1 if reg_date == ""
capture replace reg_date = v_df_017_sf_f_rep_1 if reg_date == ""
capture replace reg_date = v_df_017_sf_h_rep_1 if reg_date == ""
capture replace reg_date = v_df_779_sf_q_rep_1 if reg_date == ""
capture replace reg_date = v_df_017_sf_g_rep_1 if reg_date == ""
capture drop v_df_017_sf_d_rep_*
capture drop v_df_017_sf_f_rep_*
capture drop v_df_017_sf_h_rep_*
capture drop v_df_779_sf_q_rep_*
capture drop v_df_017_sf_g_rep_*
capture rename v_df_263_sf_c_rep_1 projected_pub_date
capture drop v_df_263_sf_c_rep_*

gen creation_date = ""
foreach i in "df_046_sf_c" "df_046_sf_k" "df_046_sf_m" "df_263_sf_b" "df_269_sf_c" "df_779_sf_n" {
	capture replace creation_date = v_`i'_rep_1 if creation_date == ""
	capture drop v_`i'*
}

gen pub_date = ""
foreach i in "df_269_sf_d" "df_263_sf_a" "df_779_sf_o"  {
	capture replace pub_date = v_`i'_rep_1 if pub_date == ""
	capture drop v_`i'*
}

capture confirm variable v_df_349_sf_a_rep_1
if _rc == 0 {
	gen type_pre_reg_work = v_df_349_sf_a_rep_1	
	forvalues i = 2/6 {
		replace type_pre_reg_work = type_pre_reg_work + " & " +  v_df_349_sf_a_rep_`i' if v_df_349_sf_a_rep_`i' != ""
	}
	dis "f"
}

capture drop v_df_349_sf_a_rep_*
capture rename v_df_591_sf_a_rep_1 desc_of_pre_reg_work
capture drop v_df_591_sf_a_rep_*