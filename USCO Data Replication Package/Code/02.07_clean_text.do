********************************************************************************
* Clean text                                                                   *
* Last modified April 10, 2024                                                 *
********************************************************************************

replace reg_num = trim(subinstr(reg_num, "/", "", 3))
replace reg_date = reg_date + "01-01" if length(reg_date) == 4
capture drop temp
gen temp = date(reg_date, "YMD")
format temp %td
drop reg_date
rename temp reg_date
order reg_date , after(reg_num)
destring creation_date, replace force
replace publication_stat = "" if publication_stat != "UNP" & publication_stat != "PUB"
sencode publication_stat, gsort(publication_stat) replace
replace pub_date = pub_date + "01-01" if length(pub_date) == 4
capture drop temp
gen temp = date(pub_date, "YMD")
format temp %td
drop pub_date
rename temp pub_date
order pub_date , after(place_of_1st_pub)
destring num_authors, replace
replace num_authors = 0 if num_authors == .
destring num_claimants, replace
replace num_claimants = 0 if num_claimants == .
expand 2 if _n == 1, gen(new)

foreach i of varlist *_corp_ind {
	replace `i' = "corp" if new == 1
	sencode `i', replace gsort(`i') fast
}
drop if new == 1 
drop new

foreach i of varlist author_*_name author_*_citizanship author_*_domicile author_*_wfh claimant_*_name {
	replace `i' = trim(lower(`i'))
}

foreach i of varlist *citizanship {	
	replace `i' = regexr(`i', "\.|\;|\:|\,", "")
	replace `i' = regexr(`i', "\.|\;|\:|\,", "")
	replace `i' = trim(subinstr(`i', "citizenship", "" , 1))
}

foreach i of varlist *domicile {	
	replace `i' = regexr(`i', "\.|\;|\:|\,", "")
	replace `i' = regexr(`i', "\.|\;|\:|\,", "")
	replace `i' = trim(subinstr(`i', "domicile", "" , 1))
}

foreach i of varlist *wfh {	
	replace `i' = "employer for hire" if regexm(`i', "for hire" )
	replace `i' = "" if `i' != "employer for hire"
	sencode `i', gsort(`i') replace fast
}

forvalues i = 1/10 {
	capture confirm variable author_`i'_brth_dth
	if _rc == 0 {
		dropmiss author_`i'_brth_dth , force
		capture confirm variable author_`i'_brth_dth
		if _rc == 0 {
			split author_`i'_brth_dth, p("-") l(2)
			replace author_`i'_brth_dth1 = trim(author_`i'_brth_dth1)
			destring author_`i'_brth_dth1, replace force
			rename author_`i'_brth_dth1 author_`i'_brth_year
			order author_`i'_brth_year, after(author_`i'_name)
			capture confirm variable author_`i'_brth_dth2
			if _rc == 0 {
				replace author_`i'_brth_dth2 = trim(author_`i'_brth_dth2)
				destring author_`i'_brth_dth2, replace force
				rename author_`i'_brth_dth2 author_`i'_dth_year
				order author_`i'_dth_year, after(author_`i'_brth_year)
			}
			drop author_`i'_brth_dth
		}
	}
}

capture drop flag
gen flag = 0
foreach i of varlist * {
	capture confirm string variable `i' 
	if _rc == 0 {
		replace flag = 1 if regexm(`i', "&")
	}
}

qui sum flag
if r(max) == 1 {
	capture frame drop amp
	frame put if flag == 1, into(amp)
	drop if flag == 1
	frame change amp
	foreach i of varlist * {
		capture confirm string variable `i' 
		if _rc == 0 {
			replace `i' = subinstr(`i', "&amp;", "&", .)
			replace `i' = subinstr(`i', "&amp", "&", .)
			replace `i' = subinstr(`i', "&apos;", "'", .)
			replace `i' = subinstr(`i', "&quot;", `"""', .)
			replace `i' = subinstr(`i', "&gt;", ">", .)
			replace `i' = subinstr(`i', "&lt;", "<", .)
			
			
		}
	}
	frame change default
	frameappend amp
	frame drop amp

}
drop flag
sort reg_num
replace work_type = trim(substr(lower(work_type), 3, .))