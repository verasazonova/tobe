from tobe.corpus import mask_paragraph


def test_mask_paragraph(nlp):

    with open('tests/text.txt') as ftext, open('tests/masked_text.txt') as fmasked:
        for line, masked_line in zip(ftext, fmasked):
            paragraph = line.strip()
            paragraph_tokens = nlp(paragraph)

            masked = ' '.join(mask_paragraph(paragraph_tokens, 6))

            masked_gold = masked_line.strip()

            print()
            print(masked)

            print(masked_gold)

            #assert masked == masked_gold
