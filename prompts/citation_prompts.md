# Citation Prompts

## Citation Instruction Prompt

This prompt ensures the LLM provides proper citations:

```
CITATION REQUIREMENTS:
- Every factual claim MUST have a citation
- Use format: [Source: DocumentName, Page X]
- Citations should appear immediately after the relevant statement
- Multiple sources can be cited: [Source: Doc1, Page 2] [Source: Doc2, Page 5]

EXAMPLE:
"The company reported revenue of $5.2 billion [Source: Annual_Report_2023, Page 15], 
which represents a 12% increase from the previous year [Source: Annual_Report_2023, Page 16]."
```

## Citation Validation Prompt

Used in post-processing to validate citations:

```python
CITATION_PATTERN = r'\[Source:\s*([^,\]]+)(?:,\s*Page\s*(\d+))?\]'
```

## Citation Format Examples

### Valid Formats:
- `[Source: HBL_Credit_Card_Terms, Page 5]`
- `[Source: Annual_Report]` (page optional)
- `[Source: Financial_Statements, Page 12]`

### Invalid Formats (to avoid):
- `(Source: Document)` - wrong brackets
- `[doc: name]` - wrong keyword
- `Source: Document` - no brackets

## Multi-Source Citation

```
When information comes from multiple sources:
"Interest rates range from 15% to 24% [Source: Rate_Schedule, Page 1] 
depending on credit score [Source: Eligibility_Criteria, Page 3]."
```

---

## Regex Patterns Used

```python
# Primary pattern - strict format
r'\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]'

# Secondary pattern - optional page
r'\[Source:\s*([^,\]]+)(?:,\s*Page\s*(\d+))?\]'

# Alternative patterns for flexibility
r'\(Source:\s*([^,\)]+)(?:,\s*Page\s*(\d+))?\)'
r'\[([^,\]]+),\s*Page\s*(\d+)\]'
```
