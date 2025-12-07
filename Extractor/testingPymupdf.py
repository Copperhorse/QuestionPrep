import pprint

import pymupdf
import pymupdf.layout
import pymupdf4llm

doc = pymupdf.open(filename="/home/copper/Documents/2510.04871v1.pdf")
pprint.pp(doc.metadata)
pprint.pp(doc._name)

markdown = pymupdf4llm.to_markdown(
    doc=doc,
    header=False,
    footer=False,
)
pprint.pp(markdown)
print(markdown)
