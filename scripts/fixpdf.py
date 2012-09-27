# Fix PDF files left corrupted by failing to close them in matplotlib
# (e.g. if an exception occurs before pp.close() is called).


import re
import sys, os

font_magic="""
3 0 obj
<< /F1 %d 0 R >>
endobj
%d 0 obj
<<
/Type /Font
/Subtype /Type1
/Name /F1
/BaseFont /Helvetica
/Encoding /WinAnsiEncoding
>>
"""

fnames = sys.argv[1:]

for fname in fnames:
    f = open(fname, 'r')
    s = f.read()
    f.close()
#    i = re.finditer("(\d+ \d+) obj\n<< .+?/Parent 2 0 R.+?>>", s, flags=re.DOTALL)
    i = re.finditer("(\d+ \d+) obj\n.+?/Parent 2 0 R", s)
    pagestrs = ["%s R" % (match.group(1),) for match in i]

    objs = [int(x) for x in re.findall("(\d+) 0 obj\n", s)]
    next_obj = max(objs)+1
    font_str = font_magic % (next_obj,next_obj)
    trailer_str = "2 0 obj\n << /Count %d /Kids [ %s ] /Type /Pages >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%%%EOF" % (len(pagestrs), " ".join(pagestrs))
    new_s = s + font_str + trailer_str
#    fn, fe = os.path.splitext(fname)
    new_fname = fname
    f = open(new_fname, 'w')
    f.write(new_s)
    print "found %d pages, writing output to %s" % (len(pagestrs), new_fname)
    f.close()
