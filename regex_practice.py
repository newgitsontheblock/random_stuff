import numpy as np
import pandas as pd
import functools
import itertools
import sys, os, re

### regex

# re.split
# split string by regex
# capturing groups are always returned with the splitted strings
# Series.str.split for DataFrames

re.split('[a-z]+', '11G22sYt33', flags=re.IGNORECASE)
re.split('([a-z]+)', '11G22sYt33', flags=re.IGNORECASE)
np.array([re.split(',', x) for x in ['dyus,8s', 'nsi,89', 'dbuy,8ss8']])

# re.findall
# returns all matched non-overlapping patterns
# has no group method, specific values can be indexed
# returns list even with just one match

re.findall('\d+', '11df')
re.findall('\d+', '11df22HsDdi33')
re.findall('\d+', '11df22HsDdi33')[0]

# re.finditer

# re.match
# returns match object
# only returns match at the beginning of string, basically re.search(^regex, string)
# only the first match will be considered
# use group to return matched value
# only use for to match pattern at beginning of string, then slightly faster than re.search

re.match('\d+', '55uydw68u')
re.match('\d+', 'uydw68u')
re.match('\d+', '55uydw68u').group(0)

# re.search
# returns match object
# returns matches anywhere in the string (unlike re.match), usually preferable to re.match
# only the first match will be considered

re.search('\d+', '55uydw68u')
re.search('\d+', 'uydw68u')
re.search('\d+', 'uydw68u').group(0)
[string if re.search('\d+', string) else np.nan for string in ['abc', 'abcd123', '456']]

# re.sub
# can be used to replace matched patterns, but also extract and reorder them
# when working with DataFrame one can use DataFrame.replace or Series.str.replace

re.sub('(\d+)', 'AAA','11ufi22k33')
re.sub('(\d+).*?(\d+).*?(\d+)', '\\1-\\2-\\3','11ufi22k33')

# re.compile
# use if regex is going to be reused
# rgx.match('343mmjh2') is the same as re.match('\d+', '343mmjh2')

rgx = re.compile('\d+')
rgx.match('343mmjh2')

# convert units based on suffix and make numeric

rawunits = ['12k', '54m', '660', 'ABC', '789.5M', '564.1K']

unit_dict = {'k':1e3, 'K':1e3, 'm':1e-3, 'M':1e6}

convertedunits = []

for i in rawunits:
    skip = False
    try:
        val = float(re.search('[0-9.]+', i).group(0))
        unit = re.search('(?<=[0-9])[A-z]$', i).group(0)
    except AttributeError:
        skip = True
    if skip:
        convertedunits.append(np.nan)
    else:
        convertedunits.append(val * unit_dict[unit])

# clean strings
# split strings into three numeric values
# the first two numeric values should be separated by '|' and appear before '/'
# the third numeric value should be the first value after the first '/' (till next '|')
# write results in numpy array

messystrings = ['56|87/98|56', '67|123|891/89|278|3782/728', '78|AbC|/78', '78/911|389']

cleanedstrings = []

for i in messystrings:
    patterncheck = re.search('[0-9]+\|[0-9]+.*?/[0-9]+', i)

    if patterncheck is None:
        cleanedstrings.append([np.nan] * 3)
    else:
        cleanedstrings.append(list(re.findall('([0-9]+)\|([0-9]+).*?/([0-9]+)', i)[0]))

cleanedstrings = np.array(cleanedstrings)


pd.DataFrame(np.c_[np.array(messystrings), cleanedstrings], columns=['original', 'val1', 'val2', 'val3'])