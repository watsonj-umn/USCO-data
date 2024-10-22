********************************************************************************
* Tighty up remaining datafields                                               *
* Last modified April 10, 2024                                                 *
********************************************************************************

capture confirm variable v_df_246_sf_a_rep_1
if _rc == 0 {
	replace title = v_df_246_sf_a_rep_1 if title == "" & v_df_246_sf_a_rep_1 != ""
	capture drop v_df_246_sf_a_rep_*
}

capture confirm variable v_df_500_sf_a_rep_1
if _rc == 0 {
	capture drop biblio_note
	rename v_df_500_sf_a_rep_1 biblio_note
	
	forvalues i = 2/10 {
		capture confirm variable v_df_500_sf_a_rep_`i'
		if _rc == 0 {
			replace biblio_note = biblio_note + "; " + v_df_500_sf_a_rep_`i' if v_df_500_sf_a_rep_`i' != "" & biblio_note != ""
		}
	}
	capture drop v_df_500_sf_a_rep*
	order biblio_note, after(title)
}

capture confirm variable v_df_740_sf_a_rep_1
if _rc == 0 {
	capture drop alt_title
	rename v_df_740_sf_a_rep_1 alt_title
	
	forvalues i = 2/10 {
		capture confirm variable v_df_740_sf_a_rep_`i'
		if _rc == 0 {
			replace alt_title = alt_title + "; " + v_df_740_sf_a_rep_`i' if v_df_740_sf_a_rep_`i' != "" & alt_title != ""
		}
	}
	capture drop v_df_740_sf_a_rep_*
	order alt_title, after(title)
}

capture confirm variable v_df_245_sf_c_rep_1
if _rc == 0 {
	rename  v_df_245_sf_c_rep_1 state_of_responsiblity
	capture drop v_df_245_sf_c_rep_*
	capture order state_of_responsiblity , after(title)
	capture order state_of_responsiblity , after(alt_title)
	capture order state_of_responsiblity , after(biblio_note)
}

rename place_of_1st_pub pub_country
capture confirm variable v_df_260_sf_a_rep_1
if _rc == 0 {
	foreach i of varlist v_df_260_sf_a_rep_* {
		replace `i' = subinstr(`i', ":","", .)
		replace `i' = trim(subinstr(`i', ";","", .))
	}
	rename v_df_260_sf_a_rep_1 pub_city
	forvalues i = 2/10 {
		capture confirm variable v_df_260_sf_a_rep_`i'
		if _rc == 0 {
			replace pub_city = pub_city + "; " + v_df_260_sf_a_rep_`i' if v_df_260_sf_a_rep_`i' != "" & pub_city != ""
		}
	}
	capture order pub_city , after(publication_stat)
	capture order pub_city , after(pub_country)
	capture drop v_df_260_sf_a_rep_*
}

capture confirm variable v_df_260_sf_b_rep_1
if _rc == 0 {
	foreach i of varlist v_df_260_sf_b_rep_* {
		replace `i' = subinstr(`i', ":","", .)
		replace `i' = trim(subinstr(`i', ";","", .))
	}
	rename v_df_260_sf_b_rep_1 publisher
	forvalues i = 2/10 {
		capture confirm variable v_df_260_sf_b_rep_`i'
		if _rc == 0 {
			replace publisher = publisher + "; " + v_df_260_sf_b_rep_`i' if v_df_260_sf_b_rep_`i' != "" & publisher != ""
		}
	}
	capture drop v_df_260_sf_b_rep_*
	capture order publisher , after(publication_stat)
	capture order publisher , after(pub_country)
	capture order publisher , after(pub_city)
}

capture confirm variable v_df_260_sf_c_rep_1
if _rc == 0 {
	replace v_df_260_sf_c_rep_1 = regexs(1) if regexm(v_df_260_sf_c_rep_1, "([0-9][0-9][0-9][0-9])")
	replace v_df_260_sf_c_rep_1 = "" if !regexm(v_df_260_sf_c_rep_1, "([0-9][0-9][0-9][0-9])")
	replace v_df_260_sf_c_rep_1 = v_df_260_sf_c_rep_1 + "-01-01" if v_df_260_sf_c_rep_1 != ""
	capture drop temp
	gen temp = date(v_df_260_sf_c_rep_1, "YMD")
	format temp %td
	replace pub_date = temp if pub_date == .
	drop v_df_260_sf_c_rep_* temp
}
order pub_date, after(publication_stat)

capture confirm variable v_df_597_sf_a_rep_1
if _rc == 0 {
	foreach i of varlist v_df_597_sf_a_rep_* {
		replace `i' = regexr(`i', "(\.|:)$", "")
		replace `i' = trim(`i')
	}
	rename v_df_597_sf_a_rep_1 previous_reg_note
	forvalues i = 2/10 {
		capture confirm variable v_df_597_sf_a_rep_`i'
		if _rc == 0 {
			replace previous_reg_note = previous_reg_note + "; " + v_df_597_sf_a_rep_`i' if v_df_597_sf_a_rep_`i' != "" & previous_reg_note != ""
		}
	}
	capture drop v_df_597_sf_a_rep_*
	capture order previous_reg_note , after(reg_date)
	capture order previous_reg_note , after(creation_date)
}
capture drop v_df_508_sf_a_rep_*

forvalues i = 1/10 {
	capture confirm variable v_df_249_sf_d_rep_`i'
	if _rc == 0 {
		split v_df_249_sf_d_rep_`i', p("-") l(2)
		destring v_df_249_sf_d_rep_`i'1 , replace force
		capture destring v_df_249_sf_d_rep_`i'2 , replace force
		rename v_df_249_sf_d_rep_`i'1 claimant_`i'_brth_year
		capture rename v_df_249_sf_d_rep_`i'2 claimant_`i'_dth_year
		capture order claimant_`i'_brth_year, after(claimant_`i'_name)
		capture order claimant_`i'_brth_year, after(claimant_`i'_corp_ind)
		capture order claimant_`i'_brth_year, after(claimant_`i'_address)
		capture order claimant_`i'_dth_year, after(claimant_`i'_name)
		capture order claimant_`i'_dth_year, after(claimant_`i'_corp_ind)
		capture order claimant_`i'_dth_year, after(claimant_`i'_address)
		capture order claimant_`i'_dth_year, after(claimant_`i'_brth_year)
		capture drop v_df_249_sf_d_rep_*
	}
}

capture confirm variable v_df_269_sf_e_rep_1
if _rc == 0 {
	capture drop temp
	gen temp = date(v_df_269_sf_e_rep_1, "DM20Y") 
	format temp %td
	replace creation_date = temp
	rename creation_date fst_commercialized
	capture drop temp v_df_269_sf_e_rep_*
	format fst_commercialized %td
}

capture confirm variable v_df_440_sf_a_rep_1
if _rc == 0 {
	rename v_df_440_sf_a_rep_1 series_name
	capture confirm variable v_df_440_sf_a_rep_2
	if _rc == 0 {
		replace series_name = series_name + ": " + v_df_440_sf_a_rep_2
	}
	capture drop v_df_440_sf_a_rep_*
	capture order series_name, after(title)
	capture order series_name, after(alt_title)
}

capture confirm variable v_df_440_sf_v_rep_1
if _rc == 0 {
	rename v_df_440_sf_v_rep_1 volume_num
	capture confirm variable v_df_440_sf_v_rep_2
	if _rc == 0 {
		replace volume_num = volume_num + "; " + v_df_440_sf_v_rep_2 if volume_num != "" & v_df_440_sf_v_rep_2 != ""
	}
	capture drop v_df_440_sf_v_rep_*
	capture order volume_num, after(title)
	capture order volume_num, after(alt_title)
	capture order volume_num, after(series_name)
}

capture confirm variable v_df_022_sf_a_rep_1
if _rc == 0 {
	foreach i of varlist v_df_022_sf_a_rep_* {
		replace `i' = regexr(`i', "(\.|:)$", "")
		replace `i' = trim(`i')
	}
	rename v_df_022_sf_a_rep_1 issn_num
	forvalues i = 2/10 {
		capture confirm variable v_df_022_sf_a_rep_`i'
		if _rc == 0 {
			replace issn_num = issn_num + "; " + v_df_022_sf_a_rep_`i' if v_df_022_sf_a_rep_`i' != "" & issn_num != ""
		}
	}
	capture drop v_df_022_sf_a_rep_*
	capture order issn_num , after(title)
	capture order issn_num , after(alt_title)
	capture order issn_num , after(series_name)
	capture order issn_num , after(volume_num)
}

capture confirm variable v_df_310_sf_a_rep_1
if _rc == 0 {
	rename v_df_310_sf_a_rep_1 periodical_freq
	foreach i in "title" "alt_title" "series_name" "volume_num" {
		capture order periodical_freq, after(`i')		
	}
	capture drop v_df_310_sf_a_rep_*
}

capture confirm variable v_df_779_sf_b_rep_1
if _rc == 0 {
	foreach i of varlist v_df_779_sf_b_rep_* {
		replace `i' = regexr(`i', "(\.|:)$", "")
		replace `i' = trim(`i')
	}
	rename v_df_779_sf_b_rep_1 temp
	forvalues i = 2/10 {
		capture confirm variable v_df_779_sf_b_rep_`i'
		if _rc == 0 {
			replace temp = temp + "; " + v_df_779_sf_b_rep_`i' if v_df_779_sf_b_rep_`i' != "" & temp != ""
		}
	}
	capture drop v_df_779_sf_b_rep_*
	capture confirm variable volume_num
	if _rc != 0 {
		gen volume_num = ""
	}
	replace volume_num = temp if temp != ""
	capture order volume_num, after(title)
	capture order volume_num, after(alt_title)
	capture order volume_num, after(series_name)
	capture drop temp

}

forvalues i = 1/30 {
	capture confirm variable v_df_505_sf_t_rep_`i'
	if _rc == 0 {
		rename v_df_505_sf_t_rep_`i' track_title_`i'
		order track_title_`i' , last
	}
}

capture drop v_df_028_sf_b_rep_*
capture confirm variable v_df_020_sf_a_rep_1
if _rc == 0 {
	local j = 1
	foreach i of varlist v_df_020_sf_a_rep_* {
		rename `i' isbn_`j'
		local j = `j' + 1
	}
	forvalues i = 1/10 {
		local j = 11 - `i'
		capture order isbn_`j' , after(biblio_note)
	}
}

foreach i of varlist * {
	label variable `i' ""
}
capture gsort -reg_date -creation_date
capture gsort -reg_date -fst_commercialized
capture drop v_df_279_sf_p*