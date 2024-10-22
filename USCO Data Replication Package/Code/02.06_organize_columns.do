********************************************************************************
* Organize datafields                                                          *
* Last modified April 10, 2024                                                 *
********************************************************************************

capture drop row_id
order reg_num	reg_date	creation_date	publication_stat	place_of_1st_pub	pub_date	title	work_type	scope_of_rights	permissions_contact	num_authors	num_claimants
capture order reg_num	reg_date	creation_date	publication_stat	projected_pub_date place_of_1st_pub	pub_date	title	work_type	type_pre_reg_work desc_of_pre_reg_work scope_of_rights	permissions_contact	num_authors	num_claimants

capture confirm variable author_1_citizanship
if _rc != 0 {
	gen author_1_citizanship = ""
}
capture confirm variable author_1_domicile
if _rc != 0 {
	gen author_1_domicile = ""
}
capture confirm variable author_1_name
if _rc != 0 {
	gen author_1_name = ""
}
capture confirm variable claimant_1_name
if _rc != 0 {
	gen claimant_1_name = ""
}
capture confirm variable author_1_wfh
if _rc != 0 {
	gen author_1_wfh = ""
}

forvalues i = 1/10 {
	local j = 11 - `i'
	capture order claimant_`j'_trans_stat , after(num_claimants)
	capture order claimant_`j'_address , after(num_claimants)
	capture order claimant_`j'_corp_ind , after(num_claimants)
	capture order claimant_`j'_name , after(num_claimants)
}

forvalues i = 1/10 {
	local j = 11 - `i'
	capture order author_`j'_auth_stat , after(num_claimants)
	capture order author_`j'_wfh , after(num_claimants)
	capture order author_`j'_brth_dth , after(num_claimants)
	capture order author_`j'_domicile , after(num_claimants)
	capture order author_`j'_citizanship , after(num_claimants)	
	capture order author_`j'_corp_ind , after(num_claimants)
	capture order author_`j'_name , after(num_claimants)
}