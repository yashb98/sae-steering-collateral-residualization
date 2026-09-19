"""Compile the verified behavioral follow-up into a standalone PDF report."""
from datetime import date
import argparse
import os
import shutil
import hashlib
import html
import json
from pathlib import Path
import subprocess
import tempfile

import markdown

ROOT = Path(__file__).resolve().parent
STEM = 'RESEARCH_UPDATE_2026-09-18'



def render(text):
    content = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    content = content.replace('<h3>gpt2_small</h3>', '<h3>GPT-2-small</h3>')
    content = content.replace('<h3>gemma_2_2b</h3>', '<h3>Gemma-2-2B</h3>')
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', help='path to a Chromium-compatible headless browser')
    args = parser.parse_args()
    browser = args.browser or os.environ.get('CHROMIUM')
    if not browser:
        cached = sorted((Path.home()/'.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-linux/headless_shell'))
        browser = str(cached[-1]) if cached else next((p for name in ['chromium', 'chromium-browser', 'google-chrome'] if (p := shutil.which(name))), None)
    if not browser:
        parser.error('Provide --browser or CHROMIUM with the path to an installed Chromium browser')
    verification = json.loads((ROOT / 'VERIFICATION.json').read_text())
    assert verification['complete'] and verification['resume_preserved']
    for setting, record in verification['settings'].items():
        for name, digest in record['hashes'].items():
            assert hashlib.sha256((ROOT / 'results' / setting / name).read_bytes()).hexdigest() == digest
    results = (ROOT / 'RESULTS.md').read_text()
    addendum = (ROOT / 'PAPER_ADDENDUM.md').read_text()
    manuscript = addendum.split('## Proposed manuscript text\n', 1)[1].split('## Evidence\n', 1)[0]
    gpt = results.split('## gpt2_small\n', 1)[1].split('## gemma_2_2b\n', 1)[0]
    gemma = results.split('## gemma_2_2b\n', 1)[1].split('## Matched comparisons\n', 1)[0]
    secondary = results.split('## Secondary blinded assessment\n', 1)[1].split('## Interpretation limits\n', 1)[0]
    limits = results.split('## Interpretation limits\n', 1)[1]
    css = '''
    @page { size: A4; margin: 17mm 17mm 16mm; @bottom-right { content: counter(page); font-size: 8pt; color: #64748b; } }
    * { box-sizing: border-box; }
    body { font: 10pt/1.43 "DejaVu Sans", sans-serif; color: #1c2838; margin: 0; }
    .page { break-before: page; }
    .page:first-child { break-before: auto; }
    h1 { font-size: 23pt; line-height: 1.15; margin: 0 0 8pt; color: #163b60; }
    h2 { font-size: 17pt; line-height: 1.2; margin: 0 0 12pt; color: #163b60; }
    h3 { font-size: 12pt; margin: 15pt 0 6pt; color: #163b60; }
    p { margin: 7pt 0; }
    a { color: #245c8e; text-decoration: none; overflow-wrap: anywhere; }
    .meta { font-size: 8.5pt; color: #64748b; margin-bottom: 13pt; }
    .notice { background: #eef4f8; border-left: 3pt solid #245c8e; padding: 8pt 10pt; margin: 12pt 0; }
    table { border-collapse: collapse; table-layout: fixed; width: 100%; margin: 11pt 0; font-size: 8pt; line-height: 1.35; break-inside: avoid; }
    th { background: #eaf0f5; color: #163b60; font-weight: bold; }
    th, td { padding: 6pt 4pt; border-bottom: .6pt solid #ccd6e0; overflow-wrap: anywhere; }
    th:first-child, td:first-child { text-align: left !important; }
    tbody tr:nth-child(even) { background: #f7f9fb; }
    thead { display: table-header-group; }
    img { max-width: 100%; height: auto; }
    pre { font-size: 8.5pt; white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f6f8; padding: 10pt; }
    li { margin-bottom: 5pt; }
    h2, h3 { break-after: avoid; }
    p, li { orphans: 3; widows: 3; }
    .manuscript { font-size: 9.2pt; line-height: 1.38; }
    .manuscript h3 { font-size: 11pt; margin-top: 12pt; }
    .caption { font-size: 9pt; color: #475569; }
    '''
    sections = [
        '<section class="page manuscript"><h1>Behavioral transfer of collateral scores</h1>'
        '<div class="meta">Research update for arXiv:2606.08365<br>'
        f'Experiments completed 18 September 2026 · PDF compiled {html.escape(date.today().isoformat())}</div>'
        '<div class="notice"><strong>Status:</strong> supplementary results and proposed manuscript text. '
        'Full manuscript integration is pending. Neither model meets the joint behavioral-transfer criterion.</div>'
        '<h3>Proposed manuscript addition</h3>' + render(manuscript) + '</section>',
        '<section class="page"><h2>GPT-2-small: held-out results</h2>' + render(gpt) + '</section>',
        '<section class="page"><h2>Gemma-2-2B: held-out results</h2>' + render(gemma) + '</section>',
        '<section class="page"><h2>Matched behavioral comparisons</h2>'
        '<img src="results/figures/behavioral_contrasts.png" alt="Matched low-minus-high behavioral comparisons">'
        '<p class="caption">Pointwise 95% bootstrap intervals resample matched feature pairs and prompt identities. '
        'Review seeds stay together. Factual correctness is identical across all arms on these prompts.</p>'
        '<h3>Interpretation</h3><p>The estimated sentiment difference is +4.2 percentage points in GPT-2 '
        '(95% CI -0.8 to +9.2) and +0.6 in Gemma (-2.4 to +3.9). Both satisfy the exploratory -5 point '
        'noninferiority margin; neither shows an advantage in factual preservation. Zero-width factual '
        'difference intervals reflect identical observed outcomes and do not establish general equivalence.</p>'
        '</section>',
        '<section class="page"><h2>Secondary blinded assessment</h2>' + render(secondary) + '</section>',
        '<section class="page"><h2>Interpretation limits and reproduction</h2>' + render(limits)
        + '<h3>Verification and source artifacts</h3><p>Both setting audits pass. The saved experiment contains '
        '7,296 held-out outputs and 456 secondary criterion judgments on 152 fixed continuations. '
        'Seven regression tests passed on 18 September. Predictor reconstruction errors are below 4 × 10<sup>-15</sup>; '
        'all feature ranks agree exactly.</p><p>Source files: RESULTS.md, PAPER_ADDENDUM.md, VERIFICATION.json, '
        'PROTOCOL.md, SECONDARY_JUDGE_PROTOCOL.md, AMENDMENTS.md and STATUS_2026-09-18.md.</p></section>',
    ]
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Behavioral transfer: September 18 research update</title><style>' + css + '</style></head><body>' + '\n'.join(sections) + '</body></html>'
    html_path = ROOT / f'{STEM}.html'
    pdf_path = ROOT / f'{STEM}.pdf'
    html_path.write_text(document)
    with tempfile.TemporaryDirectory(prefix='duan-pdf-browser-') as profile:
        subprocess.run([str(browser), '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking',
                        '--no-pdf-header-footer', f'--user-data-dir={profile}', f'--print-to-pdf={pdf_path}',
                        html_path.as_uri()], check=True, capture_output=True, text=True, timeout=90)
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = '\n'.join(page.extract_text() for page in reader.pages)
    for required in ['GPT-2-small', 'Gemma-2-2B', 'Secondary blinded assessment', '7,296', 'Full manuscript integration']:
        assert required in text, required
    print(f'{pdf_path}\n{len(reader.pages)} pages; {pdf_path.stat().st_size:,} bytes; required sections verified')


if __name__ == '__main__':
    main()
