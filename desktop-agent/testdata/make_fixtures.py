#!/usr/bin/env python3
"""Regenerates the test fixtures in this folder (vi.pdf, en.pdf, scan.pdf, hd.docx, dt.xlsx, hd_lo.pdf).
Needs:  pip install reportlab python-docx openpyxl   and (for hd_lo.pdf) LibreOffice (soffice) + DejaVu fonts.
Run from this folder:  python3 make_fixtures.py
"""
import os, shutil, subprocess

# --- PDFs made with ReportLab (embedded Unicode TTF -> clean Vietnamese text)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('DV', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
c = canvas.Canvas('vi.pdf'); c.setFont('DV', 14); c.drawString(72, 780, 'BÁO CÁO DOANH THU QUÝ 1'); c.setFont('DV', 11)
for i, l in enumerate(['Tổng doanh thu: 1.250.000.000 đồng', 'Khách hàng mới: Nguyễn Văn Ánh, Trần Thị Bích', 'Ghi chú: hoá đơn số HD-2026-001 đã thanh toán']):
    c.drawString(72, 740 - i * 18, l)
c.showPage(); c.setFont('DV', 11); c.drawString(72, 780, 'Trang hai: kết luận – tăng trưởng 12%'); c.save()
c = canvas.Canvas('en.pdf'); c.setFont('Helvetica', 12); c.drawString(72, 780, 'Hello PDF world. Invoice 42 total 100 USD.'); c.save()
c = canvas.Canvas('scan.pdf'); c.rect(100, 100, 200, 200); c.save()          # no text layer at all (like a scan)

# --- Word with a table
import docx
d = docx.Document(); d.add_heading('Hợp đồng dịch vụ', 1); d.add_paragraph('Bên A: Công ty ABC. Bên B: Nguyễn Văn Ánh.')
t = d.add_table(rows=2, cols=3)
for j, v in enumerate(['Hạng mục', 'SL', 'Giá']): t.cell(0, j).text = v
for j, v in enumerate(['Máy chủ', '2', '15.000.000']): t.cell(1, j).text = v
d.add_paragraph('Điều khoản thanh toán: 30 ngày.'); d.save('hd.docx')

# --- Excel: 2 sheets, number, bool, formula without cached value, special characters
import openpyxl
wb = openpyxl.Workbook(); ws = wb.active; ws.title = 'Doanh thu'
ws.append(['Tháng', 'Doanh thu', 'Ghi chú']); ws.append(['T1', 100.5, 'tốt']); ws.append(['T2', 200, None]); ws['E2'] = '=B2+B3'
ws.append(['T3', True, 'có "dấu" & <tag>'])
w2 = wb.create_sheet('Khách hàng'); w2.append(['Tên', 'SĐT']); w2.append(['Nguyễn Văn Ánh', '0901234567'])
wb.save('dt.xlsx')

# --- A "real world" PDF (LibreOffice export of the docx: Carlito/Caladea fonts, composed Vietnamese marks)
if shutil.which('soffice'):
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', 'hd.docx'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace('hd.pdf', 'hd_lo.pdf')
print('fixtures written')
