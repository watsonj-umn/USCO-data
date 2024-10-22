********************************************************************************
* Consuladate and clean names                                                  *
* Last modified April 10, 2024                                                 *
********************************************************************************
foreach i in "249_sf_c_rep_1" "279_sf_c_rep_1" "710_sf_a_rep_1" "700_sf_a_rep_1" {
	capture confirm variable v_df_`i'
	if _rc != 0 {
		dis "`i'"
		gen v_df_`i' = ""
	}
}

foreach i of varlist *249_sf_c* *279_sf_c* *710_sf_a* *700_sf_a* {
	replace `i' = regexr(`i', "(,)$", "")
	replace `i' = regexr(`i', "\.", "")
	replace `i' = regexr(`i', "(:)$", "")
	replace `i' = regexr(`i', "(;)", "")
	replace `i' = regexr(`i', "(\&apos)", "")
	replace `i' = trim(lower(`i'))
}

forvalues i = 1/10 {
	capture confirm variable v_df_249_sf_c_rep_`i'
	if _rc == 0 {
		capture drop corp_ind_claimant_`i'
		gen corp_ind_claimant_`i' = ""
		foreach j of varlist *700_sf_a* {
			replace corp_ind_claimant_`i' = "ind" if v_df_249_sf_c_rep_`i' == `j' & corp_ind_claimant_`i' == "" & v_df_249_sf_c_rep_`i' != ""
		}
		foreach j of varlist *710_sf_a* {
			replace corp_ind_claimant_`i' = "corp" if v_df_249_sf_c_rep_`i' == `j' & corp_ind_claimant_`i' == "" & v_df_249_sf_c_rep_`i' != ""
		}
	}
}

forvalues i = 1/10 {
	capture confirm variable v_df_279_sf_c_rep_`i'
	if _rc == 0 {
		capture drop corp_ind_auth_`i'
		gen corp_ind_auth_`i' = ""
		foreach j of varlist *700_sf_a* {
			replace corp_ind_auth_`i' = "ind" if v_df_279_sf_c_rep_`i' == `j' & corp_ind_auth_`i' == "" & v_df_279_sf_c_rep_`i' != ""
		}
		foreach j of varlist *710_sf_a* {
			replace corp_ind_auth_`i' = "corp" if v_df_279_sf_c_rep_`i' == `j' & corp_ind_auth_`i' == "" & v_df_279_sf_c_rep_`i' != ""
		}
	}
}

local cnt = 1
foreach j of varlist *700_sf_a* {
	capture split `j', p(",") limit(2)
	gen temp_700_`cnt' = ""
	capture replace temp_700_`cnt' = trim(`j'2 + " " + `j'1)
	capture drop `j'1 
	capture drop `j'2
	local cnt = `cnt' + 1
}

local cnt = 1
foreach j of varlist *710_sf_a* {
	capture split `j', p(",") limit(2)
	gen temp_710_`cnt' = ""
	capture replace temp_710_`cnt' = trim(`j'2 + " " + `j'1)
	capture drop `j'1 
	capture drop `j'2
	local cnt = `cnt' + 1
}

forvalues i = 1/10 {
	capture confirm variable v_df_249_sf_c_rep_`i'
	if _rc == 0 {
		foreach j of varlist temp_700* {
			replace corp_ind_claimant_`i' = "ind" if v_df_249_sf_c_rep_`i' == `j' & corp_ind_claimant_`i' == "" & v_df_249_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* {
			replace corp_ind_claimant_`i' = "corp" if v_df_249_sf_c_rep_`i' == `j' & corp_ind_claimant_`i' == "" & v_df_249_sf_c_rep_`i' != ""
		}
	}
}

forvalues i = 1/10 {
	capture confirm variable v_df_279_sf_c_rep_`i'
	if _rc == 0 {
		foreach j of varlist temp_700* {
			replace corp_ind_auth_`i' = "ind" if v_df_279_sf_c_rep_`i' == `j' & corp_ind_auth_`i' == "" & v_df_279_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* {
			replace corp_ind_auth_`i' = "corp" if v_df_279_sf_c_rep_`i' == `j' & corp_ind_auth_`i' == "" & v_df_279_sf_c_rep_`i' != ""
		}
	}
}

forvalues i = 1/10 {
	capture confirm variable v_df_249_sf_c_rep_`i'
	if _rc == 0 {
		capture split v_df_249_sf_c_rep_`i', p(",") limit(2)
		if _rc != 0 {
			gen v_df_249_sf_c_rep_`i'1 = v_df_249_sf_c_rep_`i'
		}
		capture confirm variable v_df_249_sf_c_rep_`i'2
		if _rc != 0 {
			gen  v_df_249_sf_c_rep_`i'2 = ""
		}
		replace v_df_249_sf_c_rep_`i'1 = trim( v_df_249_sf_c_rep_`i'1)
		replace v_df_249_sf_c_rep_`i'2 = trim( v_df_249_sf_c_rep_`i'2)
		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_claimant_`i' = "ind" if  ((v_df_249_sf_c_rep_`i'1 == `j' & v_df_249_sf_c_rep_`i'1 != "" ) | (v_df_249_sf_c_rep_`i'2 == `j' & v_df_249_sf_c_rep_`i'2 != "" )) & corp_ind_claimant_`i' == ""   & v_df_249_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* *710_sf_a* {
			replace corp_ind_claimant_`i' = "corp" if ((v_df_249_sf_c_rep_`i'1 == `j' & v_df_249_sf_c_rep_`i'1 != "" ) | (v_df_249_sf_c_rep_`i'2 == `j' & v_df_249_sf_c_rep_`i'2 != "" )) & corp_ind_claimant_`i' == ""   & v_df_249_sf_c_rep_`i' != ""
		}
		replace v_df_249_sf_c_rep_`i'1 = regexr( v_df_249_sf_c_rep_`i'1, "^(the)", "")
		replace v_df_249_sf_c_rep_`i'2 = regexr( v_df_249_sf_c_rep_`i'2, "^(the)", "")

		replace  v_df_249_sf_c_rep_`i'1 = trim( v_df_249_sf_c_rep_`i'1)
		replace  v_df_249_sf_c_rep_`i'2 = trim( v_df_249_sf_c_rep_`i'2)
		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_claimant_`i' = "ind" if  ((v_df_249_sf_c_rep_`i'1 == `j' & v_df_249_sf_c_rep_`i'1 != "" ) | (v_df_249_sf_c_rep_`i'2 == `j' & v_df_249_sf_c_rep_`i'2 != "" )) & corp_ind_claimant_`i' == ""   & v_df_249_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* {
			replace corp_ind_claimant_`i' = "corp" if ((v_df_249_sf_c_rep_`i'1 == `j' & v_df_249_sf_c_rep_`i'1 != "" ) | (v_df_249_sf_c_rep_`i'2 == `j' & v_df_249_sf_c_rep_`i'2 != "" )) & corp_ind_claimant_`i' == ""   & v_df_249_sf_c_rep_`i' != ""
		}
		drop v_df_249_sf_c_rep_`i'1 v_df_249_sf_c_rep_`i'2
	}
}

forvalues i = 1/10 {
	capture confirm variable v_df_279_sf_c_rep_`i'
	if _rc == 0 {
		capture split v_df_279_sf_c_rep_`i', p(",") limit(2)
		capture confirm variable v_df_279_sf_c_rep_`i'1
		if _rc != 0 {
			gen  v_df_279_sf_c_rep_`i'1 = ""
		}
		capture confirm variable v_df_279_sf_c_rep_`i'2
		if _rc != 0 {
			gen  v_df_279_sf_c_rep_`i'2 = ""
		}
		replace v_df_279_sf_c_rep_`i'1 = trim( v_df_279_sf_c_rep_`i'1)
		replace v_df_279_sf_c_rep_`i'2 = trim( v_df_279_sf_c_rep_`i'2)
		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_auth_`i' = "ind" if  ((v_df_279_sf_c_rep_`i'1 == `j' & v_df_279_sf_c_rep_`i'1 != "" ) | (v_df_279_sf_c_rep_`i'2 == `j' & v_df_279_sf_c_rep_`i'2 != "" )) & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* *710_sf_a* {
			replace corp_ind_auth_`i' = "corp" if ((v_df_279_sf_c_rep_`i'1 == `j' & v_df_279_sf_c_rep_`i'1 != "" ) | (v_df_279_sf_c_rep_`i'2 == `j' & v_df_279_sf_c_rep_`i'2 != "" )) & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		replace v_df_279_sf_c_rep_`i'1 = regexr( v_df_279_sf_c_rep_`i'1, "^(the)", "")
		replace v_df_279_sf_c_rep_`i'2 = regexr( v_df_279_sf_c_rep_`i'2, "^(the)", "")

		replace  v_df_279_sf_c_rep_`i'1 = trim( v_df_279_sf_c_rep_`i'1)
		replace  v_df_279_sf_c_rep_`i'2 = trim( v_df_279_sf_c_rep_`i'2)
		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_auth_`i' = "ind" if  ((v_df_279_sf_c_rep_`i'1 == `j' & v_df_279_sf_c_rep_`i'1 != "" ) | (v_df_279_sf_c_rep_`i'2 == `j' & v_df_279_sf_c_rep_`i'2 != "" )) & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* {
			replace corp_ind_auth_`i' = "corp" if ((v_df_279_sf_c_rep_`i'1 == `j' & v_df_279_sf_c_rep_`i'1 != "" ) | (v_df_279_sf_c_rep_`i'2 == `j' & v_df_279_sf_c_rep_`i'2 != "" )) & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		drop v_df_279_sf_c_rep_`i'1 v_df_279_sf_c_rep_`i'2
		
	}
}

** match first word
forvalues i = 1/10 {
	capture confirm variable v_df_249_sf_c_rep_`i'
	if _rc == 0 {
		gen temp = word(v_df_249_sf_c_rep_`i', 1)
		replace temp = subinstr(temp, ",", "", 1)
		replace temp = subinstr(temp, ".", "", 1)
		replace temp = subinstr(temp, ";", "", 1)
		replace temp = subinstr(temp, ":", "", 1)
		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_claimant_`i' = "ind" if  strpos(`j', temp) > 0 & temp != "" & corp_ind_claimant_`i' == ""     & v_df_249_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* *710_sf_a* {
			replace corp_ind_claimant_`i' = "corp" if strpos(`j', temp) > 0 & temp != "" &  & corp_ind_claimant_`i' == ""     & v_df_249_sf_c_rep_`i' != ""
		}
		drop temp
		gen temp = word(v_df_249_sf_c_rep_`i', 2)
		replace temp = subinstr(temp, ",", "", 1)
		replace temp = subinstr(temp, ".", "", 1)
		replace temp = subinstr(temp, ";", "", 1)
		replace temp = subinstr(temp, ":", "", 1)

		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_claimant_`i' = "ind" if  strpos(`j', temp) > 0 & temp != "" & corp_ind_claimant_`i' == ""     & v_df_249_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* *710_sf_a* {
			replace corp_ind_claimant_`i' = "corp" if strpos(`j', temp) > 0 & temp != "" &  & corp_ind_claimant_`i' == ""   & v_df_249_sf_c_rep_`i' != ""
		}
		drop temp
	}
}

forvalues i = 1/10 {
	capture confirm variable v_df_279_sf_c_rep_`i'
	if _rc == 0 {
		gen temp = word(v_df_279_sf_c_rep_`i', 1)
		replace temp = subinstr(temp, ",", "", 1)
		replace temp = subinstr(temp, ".", "", 1)
		replace temp = subinstr(temp, ";", "", 1)
		replace temp = subinstr(temp, ":", "", 1)
		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_auth_`i' = "ind" if  strpos(`j', temp) > 0 & temp != "" & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* *710_sf_a* {
			replace corp_ind_auth_`i' = "corp" if strpos(`j', temp) > 0 & temp != "" &  & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		drop temp
		gen temp = word(v_df_279_sf_c_rep_`i', 2)
		replace temp = subinstr(temp, ",", "", 1)
		replace temp = subinstr(temp, ".", "", 1)
		replace temp = subinstr(temp, ";", "", 1)
		replace temp = subinstr(temp, ":", "", 1)

		foreach j of varlist temp_700* *700_sf_a*  {
			replace corp_ind_auth_`i' = "ind" if  strpos(`j', temp) > 0 & temp != "" & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		foreach j of varlist temp_710* *710_sf_a* {
			replace corp_ind_auth_`i' = "corp" if strpos(`j', temp) > 0 & temp != "" &  & corp_ind_auth_`i' == ""   & v_df_279_sf_c_rep_`i' != ""
		}
		drop temp
	}
}

** rename
forvalues i = 1/10 {
	capture confirm variable v_df_249_sf_c_rep_`i'
	if _rc == 0 {
		rename v_df_249_sf_c_rep_`i' claimant_`i'_name
		rename corp_ind_claimant_`i' claimant_`i'_corp_ind
	}
	capture confirm variable v_df_279_sf_c_rep_`i'
	if _rc == 0 {
		rename v_df_279_sf_c_rep_`i' author_`i'_name
		rename corp_ind_auth_`i' author_`i'_corp_ind
	}
}

forvalues i = 1/10 {
	capture confirm variable v_df_249_sf_e_rep_`i'
	if _rc == 0 {
		rename v_df_249_sf_e_rep_`i' claimant_`i'_trans_stat
	}
	capture confirm variable v_df_249_sf_u_rep_`i'
	if _rc == 0 {
		rename v_df_249_sf_u_rep_`i' claimant_`i'_address
	}
	capture confirm variable v_df_249_sf_f_rep_`i'
	if _rc == 0 {
		rename v_df_249_sf_f_rep_`i' claimant_`i'_address
	}
	capture confirm variable v_df_279_sf_e_rep_`i'
	if _rc == 0 {
		rename v_df_279_sf_e_rep_`i' author_`i'_auth_stat
	}
	capture confirm variable v_df_279_sf_f_rep_`i'
	if _rc == 0 {
		rename v_df_279_sf_f_rep_`i' author_`i'_domicile
	}
	capture confirm variable v_df_279_sf_g_rep_`i'
	if _rc == 0 {
		rename v_df_279_sf_g_rep_`i' author_`i'_citizanship
	}
	capture confirm variable v_df_279_sf_h_rep_`i'
	if _rc == 0 {
		rename v_df_279_sf_h_rep_`i' author_`i'_wfh
	}
}

* assign birth and death dates
forvalues i = 1/10 {
	capture confirm variable author_`i'_name
	if _rc == 0 {
		capture drop author_`i'_brth_dth
		gen author_`i'_brth_dth = ""
		capture confirm variable v_df_279_sf_d_rep_`i'
		if _rc == 0 {
			replace author_`i'_brth_dth = v_df_279_sf_d_rep_`i'
			drop v_df_279_sf_d_rep_`i'
		}
		capture confirm variable v_df_700_sf_d_rep_`i' 
		if _rc == 0 {
			replace author_`i'_brth_dth = v_df_700_sf_d_rep_1 if author_`i'_corp_ind == "ind" & author_`i'_brth_dth == ""
		}
	}
}

capture drop temp_7*
capture drop *710_sf_*
capture drop *700_sf_*