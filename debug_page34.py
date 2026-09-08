import sys
sys.path.insert(0, "src")
from fkl.parser import parse_pdf
from fkl.extractor import build_context

d = parse_pdf(r"C:\Users\apoor\Downloads\02-delhivery-annual-report-fy24-excerpt.pdf")
ctx = build_context(d)

print("=== blocks on page 34 ===")
for i, cb in enumerate(ctx.blocks_on(34)[:6]):
    print(i, cb.role.value, cb.block.kind.value, repr(cb.block.text[:70]))

print("\n=== unit declarations on page 34 ===")
decls = ctx.unit_context.declarations_on(34)
print(f"count: {len(decls)}")
for dec in decls:
    print(f"  [{dec.char_start}:{dec.char_end}] {dec.text!r} -> {dec.unit.describe()}")

print("\n=== resolve at a mid-page position ===")
resolved, source = ctx.unit_context.resolve(34, 500)
print(f"resolved={resolved.describe() if resolved else None}, source={source}")
print(f"document_default={ctx.unit_context.document_default.describe() if ctx.unit_context.document_default else None}")