import re
s = open('/mnt/d/DVCH13-1/DVCH.tex').read()
# check braces balance
print("brace balance:", s.count('{') - s.count('}'))
# check env balance
begin = re.findall(r'\\begin\{(\w+\*?)\}', s)
end = re.findall(r'\\end\{(\w+\*?)\}', s)
from collections import Counter
b, e = Counter(begin), Counter(end)
for env in set(list(b) + list(e)):
    if b[env] != e[env]:
        print(f"ENV MISMATCH: {env}: begin={b[env]} end={e[env]}")
print("environments ok" if b == e else "see above")
# check refs to labels that exist
labels = set(re.findall(r'\\label\{([^}]+)\}', s))
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', s))
missing = refs - labels
print("missing labels referenced:", missing if missing else "none")
# check undefined citations
cites = set(re.findall(r'\\cite\{([^}]+)\}', s))
cited = set()
for c in cites:
    cited |= set(x.strip() for x in c.split(','))
bib = open('/mnt/d/DVCH13-1/refs.bib').read() + open('/mnt/d/DVCH13-1/DVCHNotes.bib').read()
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
missing_cites = cited - keys
print("citations missing from bibs:", missing_cites if missing_cites else "none")
