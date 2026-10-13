"""Synthetic, gold-free regression tests for coverage-aware alignment."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from src.e2_alignment import make_plan, validate_units, _shared_anchor_length


def opts(**kw):
    defaults=dict(alignment='semantic_coverage',chunk_tokens=16,max_unit_tokens=48,
                  embedding_model='synthetic',embedding_revision='sha-test',
                  similarity_threshold=0.46,merge_penalty=0.06,
                  coverage_weight=0.50,anchor_bonus=0.35)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class CoverageTests(unittest.TestCase):
    def test_legacy_fingerprint_unchanged(self):
        from src.e2 import fingerprint
        from types import SimpleNamespace
        old=dict(prompt='fewshot',context=20,max_tokens=512,alignment='semantic',
                 chunk_tokens=48,embedding_model='encoder',embedding_revision='sha',
                 max_unit_tokens=40,similarity_threshold=0.46,merge_penalty=0.06)
        now=old | dict(coverage_weight=0.5,anchor_bonus=0.35)
        self.assertEqual(fingerprint(SimpleNamespace(**old),'apertus','https://example/v1'),
                         fingerprint(SimpleNamespace(**now),'apertus','https://example/v1'))
        now['alignment']='semantic_coverage'
        self.assertNotEqual(fingerprint(SimpleNamespace(**old),'apertus','https://example/v1'),
                            fingerprint(SimpleNamespace(**now),'apertus','https://example/v1'))

    def test_exact_three_word_anchor(self):
        self.assertEqual(_shared_anchor_length('Links Sektion Recht Specialist staff'.split(),
                                                'Links Sektion Recht'.split()),3)
        self.assertEqual(_shared_anchor_length('Last updated 2020'.split(),
                                                'Dernière mise à jour 2020'.split()),0)

    def test_freeze_previous_plan_settings(self):
        from src.e2_alignment import alignment_settings
        x=opts(alignment='semantic')
        self.assertNotIn('coverage_weight',alignment_settings(x))
        self.assertNotIn('anchor_bonus',alignment_settings(x))
        self.assertIn('coverage_weight',alignment_settings(opts()))

    def test_continuation_covered_by_many_to_one_matching(self):
        # Sentences A1,A2 express components of B1. A3 is genuinely absent.
        # With the old reward, A1 alone wins (similarity 0.91 > merged 0.84).
        import math
        a1='A1 first subject regulation four .'
        a2='A2 list consultations publications seven .'
        a3='A3 no counterpart paragraph here .'
        b1='B1 regulations include consultations and publications in detail .'
        row=dict(id='synthetic',text_a=f'{a1} {a2} {a3}',text_b=b1)
        def v(cos):return (cos,math.sqrt(1-cos*cos),0.)
        def encode(strings):
            out=[]
            for t in strings:
                if 'B1' in t:out.append((1.,0.,0.))
                elif 'A3' in t:out.append((0.,0.,1.))
                elif 'A1' in t and 'A2' in t:out.append(v(0.84))
                elif 'A1' in t:out.append(v(0.91))
                elif 'A2' in t:out.append(v(0.86))
                else:raise ValueError(t)
            return out
        old=make_plan(row,opts(alignment='semantic'),encode)
        new=make_plan(row,opts(),encode)
        self.assertEqual([u['kind'] for u in old['units']],['matched','unaligned_a','unaligned_a'])
        self.assertEqual([u['kind'] for u in new['units']],['matched','unaligned_a'])
        self.assertEqual(new['units'][0]['ra'],[0,12])
        validate_units(new['units'],len(row['text_a'].split()),len(row['text_b'].split()),48)

    def test_anchor_without_swallowing_unrelated_addition(self):
        import math
        row=dict(id='synthetic',text_a='alpha section . For full information please select German . Links Sektion Recht Specialist staff',
                 text_b='alpha section . Links Sektion Recht')
        def encode(strings):
            out=[]
            for t in strings:
                if 'Links' in t:
                    out.append((0.,0.929,0.370) if 'Specialist' in t else (0.,0.,1.))
                elif 'For' in t:out.append((0.,1.,0.))
                elif 'alpha' in t:out.append((1.,0.,0.))
                else:raise ValueError(t)
            return out
        plan=make_plan(row,opts(),encode)
        self.assertEqual([u['kind'] for u in plan['units']],['matched','unaligned_a','matched'])
        self.assertEqual(plan['units'][-1]['rb'],[3,6])
        self.assertEqual(plan['units'][-1]['ra'][0],10)
        validate_units(plan['units'],len(row['text_a'].split()),len(row['text_b'].split()),48)

if __name__=='__main__':unittest.main()
