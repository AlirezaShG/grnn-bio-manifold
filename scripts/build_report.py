"""Generate the Persian change-rationale report from the actual release records.
Requires python-docx; run after validate and the saved development pilots.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,Cm,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'docs/GRNN_v7_Change_Report_FA.docx'
V=json.loads((ROOT/'results/validation/validation.json').read_text())
METRICS=pd.read_csv(ROOT/'results/pilot/factorial/metrics.csv')
TRAIN=pd.read_csv(ROOT/'results/pilot/factorial/training_summary.csv')
PROV=json.loads((ROOT/'results/pilot/factorial/data_provenance.json').read_text())
FACTOR=json.loads((ROOT/'results/pilot/factorial/factorial_contrasts.json').read_text())
DOC=Document()
section=DOC.sections[0]
section.page_width=Cm(21);section.page_height=Cm(29.7)
section.top_margin=Cm(1.8);section.bottom_margin=Cm(1.8)
section.left_margin=Cm(2);section.right_margin=Cm(2)
styles=DOC.styles
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
 s=styles[name];s.font.name='Noto Naskh Arabic';s.font.size=Pt(11.5)
 s._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:cs'),'Noto Naskh Arabic')
 s.paragraph_format.space_after=Pt(5)
 s.paragraph_format.line_spacing=1.15
styles['Title'].font.size=Pt(24);styles['Title'].font.bold=True
styles['Title'].font.color.rgb=RGBColor.from_string('153C55')
styles['Subtitle'].font.size=Pt(12)
for name,size in [('Heading 1',16),('Heading 2',13)]:
 styles[name].font.size=Pt(size);styles[name].font.bold=True
 styles[name].font.color.rgb=RGBColor.from_string('153C55')
 styles[name].paragraph_format.space_before=Pt(13)
 styles[name].paragraph_format.keep_with_next=True

def rtl(p):
 p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
 pr=p._p.get_or_add_pPr();el=OxmlElement('w:bidi');el.set(qn('w:val'),'1');pr.append(el)
 return p

def para(text,style=None):
 p=rtl(DOC.add_paragraph(style=style));p.add_run(text);return p

def heading(text,level=1):
 return para(text,f'Heading {level}')

def ltr(text,size=9):
 p=DOC.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.LEFT
 p.paragraph_format.line_spacing=1.1
 r=p.add_run(text);r.font.name='DejaVu Sans';r.font.size=Pt(size)
 return p

def table(headers,rows,widths=None):
 t=DOC.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False
 for i,h in enumerate(headers):
  c=t.rows[0].cells[i];c.text=str(h)
  shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'153C55');c._tc.get_or_add_tcPr().append(shade)
  for p in c.paragraphs:
   for r in p.runs:r.font.color.rgb=RGBColor(255,255,255);r.font.bold=True
 trPr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
 for row in rows:
  cells=t.add_row().cells
  for c,text in zip(cells,row):c.text=str(text)
 for row in t.rows:
  for i,c in enumerate(row.cells):
   if widths:c.width=Cm(widths[i])
   for p in c.paragraphs:
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.paragraph_format.space_after=Pt(3)
    for r in p.runs:r.font.name='DejaVu Sans';r.font.size=Pt(8)
 return t

para('بازطراحی و ارزیابی پژوهشی پروژهٔ GFR','Title')
para('گزارش جامع تغییرات نسخهٔ ۷ | بررسی منبع، روش‌ها، آزمون‌ها و مسیر مقاله','Subtitle')
para('تاریخ گزارش: ۷ نوامبر ۲۰۲۶. مبنا: آرشیو grnn_bio_manifold_project_v6_upgraded.zip و نسخهٔ مشخص‌شدهٔ ریپازیتوری AllenInstitute/GRNN.')
para('نسخهٔ ۷ یک بازسازی اجرایی و قابل‌ممیزی از طرح پژوهش است. هدف اصلی حفظ نورون‌های برازش‌شدهٔ Allen و مقایسهٔ معتبر یادگیری محلی با یادگیری مبتنی بر گرادیان است. کد، آزمون‌های عددی، آزمایش‌های توسعه‌ای اجراشده و برنامهٔ مطالعهٔ تأییدی تحویل داده شده‌اند. پذیرش ژورنال یا اعتبار زیستی نتایجی که هنوز اندازه‌گیری نشده‌اند، از آماده‌بودن نرم‌افزار نتیجه نمی‌شود.')

heading('۱. نتیجهٔ اصلی ممیزی')
para('مسئلهٔ اصلی نسخهٔ اولیه کمبود شکل یا کمبود روش تحلیل نبود؛ چند تصمیم اجرایی نتیجهٔ مقایسه را از پرسش علمی دور کرده بود. حالتی با نام «authentic» ضرایب نورون واقعی را پیش از هر آموزش تغییر می‌داد؛ بخشی از شبکه مستقیماً برای دو کانال تسک سیم‌کشی شده بود؛ محاسبه‌ای که آنلاین معرفی می‌شد زمان را به عقب طی می‌کرد؛ و بعضی آمارهای جمعیتی از همان داده‌ای استفاده می‌کردند که محور تفکیک را ساخته بود. اصلاح این موارد بر افزودن تعداد بیشتری نمودار مقدم بود.')
para('تمام داده‌ها و خروجی‌های قدیمی در پوشهٔ legacy_v6 باقی مانده‌اند و هش فایل‌های اولیه ثبت شده است. آن‌ها سابقهٔ پروژه هستند؛ شواهد عددی گزارش حاضر از اجرای نسخهٔ ۷ تولید شده‌اند. هیچ مقدار دقت، p-value یا برتری نسخهٔ قدیمی وارد جدول نتایج جدید نشده است.')
para(f'در بستهٔ تحویلی، {V["tests_run"]} آزمون خودکار با {V["failures"]} شکست و {V["errors"]} خطا اجرا شده است. پایلوت فاکتوریل نیز شامل {len(TRAIN)} اجرای واقعی، ۱۵ وضعیت و ۳ بذر شبکه است. این تعداد بذر برای نتیجه‌گیری تأییدی یا تعمیم زیستی کافی فرض نشده است.')

heading('۲. منبع Allen دقیقاً چه چیزی ارائه می‌کند؟')
para('منبع رسمی، بانک مدل‌های نرخ شلیکِ مشتق‌پذیر است که به ثبت‌های الکتروفیزیولوژی تک‌سلولی برازش شده‌اند. هر سلول ترکیبی از فیلترهای تاریخچهٔ جریان و تاریخچهٔ شلیک و یک تابع فعال‌سازی غیرخطی دارد. بنابراین مدل از لحاظ پاسخ نرخ شلیک به داده متصل است، اما شکل کامل پتانسیل عمل، مورفولوژی دندریت، سیناپس‌های رسانایی‌محور و کانکتوم همان مجموعهٔ سلول‌ها را بازسازی نمی‌کند [۱].')
para('ریپازیتوری روی commit زیر ثابت شد. فایل‌های اصلی مدل، شبکه، بارگذاری داده، پیش‌پردازش و آموزش بررسی و نسخهٔ منبع برای ممیزی در بسته نگهداری شد. دادهٔ محلی فقط پیکربندی ۲۰/۲۰ میلی‌ثانیه را دارد؛ ۱۱۲۴ رکورد آن پس از خواندن ساختار داده با زیرمجموعهٔ متناظر منبع برابر است. مقادیر متادیتای سلول‌های پیوست نیز با ردیف‌های متناظر منبع تطبیق داده شد و مغایرت مقداری مشاهده نشد.')
ltr('Upstream commit: 835a535f6831b88339c6a5c17180fdbc504f7448')
para('مقالهٔ منبع با عنوان «A dataset of differentiable biologically derived single-neuron models» در ۱۴ اکتبر ۲۰۲۶ در Cell Reports Methods منتشر شده است. اطلاعات انتشار و چکیده بررسی شده‌اند؛ بررسی فنی این بازطراحی بر کد و دادهٔ واقعاً دریافت‌شده تکیه دارد، نه ادعای خواندن متن کامل ناشر که در این محیط در دسترس مستقیم نبود [۱].')
para('یک محدودیت موروثی نیز ثبت شد: در پیش‌پردازش منبع، Noise 1 در آموزش و اعتبارسنجی برازش تک‌سلولی حضور دارد. حذف انتخاب بر اساس test_evr در پروژهٔ جدید لازم است، اما val_evr منبع را به ارزیابی فیزیولوژیک کاملاً دست‌نخورده تبدیل نمی‌کند. برای ادعای جدید دربارهٔ وفاداری به patch clamp باید پروتکل‌های خامِ واقعاً مستقل دوباره ارزیابی شوند.')

heading('۳. بازگرداندن هویت فیزیولوژیک نورون‌ها')
para('در نسخهٔ ۶، ضرایب منفی فیلترهای کند به صفر بریده می‌شدند؛ کانال سریع با ضریب ۰٫۱۶ تغییر می‌کرد؛ چند مجموعه فیلتر کند در ۵٫۵ ضرب می‌شدند؛ بخشی از فیلترهای PV حذف می‌شدند؛ و ضرایب تاریخچهٔ شلیک در ۰٫۰۸۵ ضرب می‌شدند. کنترل همگن نیز ضریب اضافی ۷٫۸ برای تاریخچه داشت. پس مقایسه، صرفاً مقایسهٔ نورون‌های برازش‌شده با کنترل آن‌ها نبود.')
para('در حالت authentic نسخهٔ ۷، a، beta، ds، آستانه، ضرایب چندجمله‌ای، مقیاس جریان و سقف نرخ شلیک همگی بدون این دست‌کاری‌ها حفظ و غیرقابل‌تغییر می‌شوند. اثر انگشت عددی این پارامترها هنگام ساخت شبکه ثبت و در طول آموزش کنترل می‌شود. علامت منفی یک ضریب تاریخچه ویژگی یک فیلتر درون‌سلولی است؛ نباید آن را با قانون دیل برای علامت سیناپس خروجی اشتباه گرفت.')
para('قرارداد واحدها صریح شد: جریان بر حسب pA، زمان بر حسب ms و نرخ شلیک بر حسب Hz است. نرخ منبع از spikes/ms فقط یک بار در ۱۰۰۰ ضرب می‌شود. ضرایب ذخیره‌شده متعلق به معادلهٔ زمان‌گسسته‌اند و نباید بار دیگر در dt ضرب شوند. معادلهٔ اجرا به شکل زیر است:')
ltr('h_t = (1 - ds) h_(t-1) + a I_t + beta r_(t-1)',10)
ltr('r_t = gamma_Hz ReLU[tanh(sum_d c_d^2 ((mean(h_t)-threshold)/max_current)^d)]',9)
para('آزمون مستقل معادله، پاسخ تمام ۱۱۲۴ سلول را با ترجمهٔ اسکالر کد رسمی، با واحد اصلی spikes/ms و تبدیل خروجی به Hz، مقایسه می‌کند. مشتق فعال‌سازی، حالت چندجمله‌ای ثابت و درجات بالاتر، نبود ضریب تکراری dt و شروع واقعاً صفر نیز آزمون دارند. این برابریِ محاسباتی جایگزین مقایسهٔ تازه با ثبت‌های آزمایشگاهی نیست.')
para('برای قرار دادن شبکه در یک نقطهٔ کاری مشخص، یک sweep جریان مستقل از تسک روی بانک دست‌نخورده انجام می‌شود. جریان ثابت نزدیک به نرخ هدف انتخاب و خطای رسیدن به آن هدف ثبت می‌شود. این جریان و مقیاس ورودی در همهٔ کنترل‌های هویتی ثابت می‌مانند. بنابراین تغییر عملکرد پس از جابه‌جایی هویت می‌تواند شامل تغییر نقطهٔ کاری نیز باشد؛ یک تحلیل حساسیت با کالیبراسیون دوباره، آزمایش جداگانه‌ای است و نباید پنهانی جایگزین کنترل اصلی شود.')

heading('۴. ماژول‌های داده و مدار')
heading('داده: grnn_bio/data.py',2)
para('فیلتر پذیرش بر اساس test_evr حذف شد. پذیرش بر اساس گونه، معیار اعتبارسنجی منبع و برچسب عملیاتی نوع سلول انجام می‌شود؛ test_evr صرفاً برای توصیف ذخیره می‌شود. نمونه‌گیری در هر شبکه بدون جایگذاری است؛ اگر تعداد سلول کافی نباشد کد خطای روشن می‌دهد. Pyr از ترکیب مورفولوژی spiny و خطوط مشخص Cre تعریف می‌شود و PV، SST و VIP برچسب‌های عملیاتی مبتنی بر Cre هستند. این برچسب‌ها اثبات شکل basket یا Martinotti و جای‌گیری لایهٔ ۱ نیستند.')
para(f'بانک واجد معیارهای چهار گروه در این اجرا شامل {sum(PROV["pool_counts"].values())} سلول است. توسعه و تأیید بر اساس specimen ID جدا شدند و اشتراک سلولی آن‌ها {PROV["development_confirmation_cell_overlap"]} است. اشتراک donor در دو مجموعه برابر {PROV["development_confirmation_donor_overlap"]} است؛ پس ادعای donor-disjoint بودن مجاز نیست. این تمایز در manifest نیز ثبت می‌شود.')
heading('مدار: grnn_bio/circuit.py',2)
para('همهٔ برش‌های ثابت مانند ۸۰:۹۰ و ۹۵:۱۰۰ با برچسب نوع سلول جایگزین شدند. مدارهای کوچک و اندازه‌های متفاوت واقعاً آزمون می‌شوند. پوشش ورودی، خروجی، بازخورد و مجموعهٔ سیناپس‌های قابل‌آموزش قابل‌ممیزی است. شنت‌های دندریتی و جسم‌سلولیِ افزوده‌شده که در برازش اصلی شناسایی نشده بودند از مدل اصلی حذف شدند. می‌توان در پژوهش بعدی آن‌ها را به عنوان فرضیهٔ مجزا اضافه و دوباره اعتبارسنجی کرد.')
para('معنای ماسک ثابت است: M[i,j] یعنی اتصال j به i. علامت خروجی هر slot بر اساس نقش تحریکی یا مهاری آن ثابت می‌ماند و قدرمطلق اتصال نامنفی است. نگاشت نرخ به جریان از یک مرجع فیزیکی مشترک ۱۰۰ Hz استفاده می‌کند؛ تقسیم بر سقف اختصاصی هر سلول حذف شد تا بخشی از ناهمگنی نرخ با نرمال‌سازی خنثی نشود و جابه‌جایی gamma خودِ قانون اتصال را تغییر ندهد.')
para('اتصالات از پیش‌فرض‌های کیفی نوع سلول الهام می‌گیرند [۲،۳]. مقادیر عددی احتمال اتصال در این پروژه دادهٔ اندازه‌گیری‌شدهٔ سلول‌های نمونه‌برداری‌شده نیستند. به همین دلیل نام و ادعای «کانکتوم واقعی» جای خود را به مدار مصنوعی با prior نوع سلول داده است. شبکهٔ کنترل با تعویض یال‌های جهت‌دار ساخته می‌شود و هم درجهٔ ورودی و هم درجهٔ خروجی، علامت دیل و مجموعهٔ وزن‌های خروجی هر نورون را حفظ می‌کند. زنجیرهٔ تعویض محدود، نمونه‌برداری یکنواخت اثبات‌شده از تمام گراف‌های مجاز تلقی نشده است.')

heading('۵. کنترل‌های هویت و طرح فاکتوریل')
para('کنترل shuffle، بردار a، بردار beta و کل tuple فعال‌سازی را مستقل از هم درون هر نوع سلول جابه‌جا می‌کند. بدین ترتیب توزیع‌های حاشیه‌ای حفظ می‌شوند اما همراهی پارامترهای متعلق به یک سلول برهم می‌خورد؛ در یک جایگشت محدود ممکن است بخشی از وابستگی باقی بماند. کنترل جابه‌جایی میان نوع‌ها یک جایگشت یک‌به‌یک از tuple کامل است و نورون‌های PV را با جایگذاری چندباره تکثیر نمی‌کند.')
para('کنترل همگن میانگین ضرایب را درون هر نوع می‌گیرد و صریحاً مصنوعی نامیده می‌شود؛ برابری میانگین ضرایب به معنای برابری gain تابع غیرخطی نیست. کنترل سریع dsهای کند را افزایش می‌دهد و a و beta را طوری مقیاس می‌کند که نسبت‌های a/ds و beta/ds حفظ شوند. این کار معادلهٔ تعادل ثابت را حفظ می‌کند، اما پاسخ گذرا و پایداری مدار را الزاماً برابر نمی‌کند. تفسیر اثر آن باید این محدودیت را در نظر بگیرد.')
para('هستهٔ اصلی یک فاکتوریل کامل ۲×۲×۲ است: هویت دست‌نخورده یا shuffle، توپولوژی مبتنی بر نوع یا بازسیم‌کشی‌شده، و BPTT یا روش محلی. بنابراین اثر متقابلِ «فیزیولوژی × توپولوژی × یادگیری» قابل محاسبه است. هفت کنترل دیگر برای حذف beta از مشتق، reservoir، همگنی، تاریخچهٔ سریع، جابه‌جایی میان نوع‌ها، perturbation و BPTT با نویز متناظر اضافه شده‌اند. این کنترل‌ها جای سلول‌های مفقود فاکتوریل را پر نمی‌کنند.')

heading('۶. بازطراحی تسک')
para('تسک از تصمیم‌گیری وابسته به context الهام گرفته، اما تکرار مستقیم یک آزمایش حیوانی نیست [۴]. پنج ورودی شامل شواهد A، شواهد B، دو کانال context و علامت go است. ترکیب context، علامت شواهد مرتبط و علامت شواهد نامرتبط متوازن است؛ در اندازه‌های batch مضرب هشت هیچ ترکیبی از قلم نمی‌افتد. همین موضوع اجازه می‌دهد عملکرد روی شواهد متعارض جدا از عملکرد سادهٔ هم‌علامت گزارش شود.')
para('تأخیر در سطح هر trial مستقل از پاسخ انتخاب می‌شود. تأخیرهای آموزش پیش‌فرض ۱۰۰، ۲۰۰ و ۳۰۰ ms و تأخیرهای جدید ۴۰۰ و ۶۰۰ ms هستند. پنجرهٔ تصمیم و padding با ماسک مستقل مشخص می‌شوند، در نتیجه طول trial به برچسب پاسخ تبدیل نمی‌شود. context زودهنگام و context پس از تأخیر دو تنظیم مستقل دارند؛ دومی نگهداری هر دو جریان حسی را تا آشکار شدن قاعده مطالبه می‌کند.')
para('افزایش تأخیر، افزایش نویز حسی و افزودن مزاحم در تأخیر در سه آزمون جدا بررسی می‌شوند و یک آزمون ترکیبی نیز وجود دارد. در نسخهٔ قبل تغییر هم‌زمان تأخیر و نویز نسبت‌دادن اثر به یک عامل را دشوار می‌کرد. منحنی psychometric برای هر دو context ساخته می‌شود. شواهد صفر برچسب اجباری مثبت ندارد؛ احتمال انتخاب روی آن قابل گزارش است، اما دقتِ دارای پاسخ صحیح ساختگی محاسبه نمی‌شود.')
para('دو کنترل میان‌بُر، context را هنگام inference حذف می‌کنند یا وزن بازگشتی را صفر می‌کنند و همان trialها را دوباره می‌سنجند. این کنترل‌ها در کنار reservoir روشن می‌کنند چه بخشی از عملکرد به recurrent learning یا قاعدهٔ context وابسته است. دقت بالای کلی وقتی trialهای متعارض ضعیف‌اند به‌تنهایی حل کامل مسئله را نشان نمی‌دهد.')

heading('۷. بازسازی سازوکارهای یادگیری')
para('تابع زیان برای همهٔ روش‌های اصلی، BCE روی تک‌تک binهای پنجرهٔ تصمیم است که ابتدا داخل هر trial و سپس در batch میانگین گرفته می‌شود. پاسخ نهایی علامت میانگین logit تصمیم است. این تعریف، استفاده از BCE روی میانگین logit یا MSE قدیمی نیست؛ کد آموزش و ارزیابی از تعریف یکسان استفاده می‌کنند.')
heading('BPTT کامل',2)
para('مشتق کاملِ تمام فیلترها، مسیر بازخورد درون‌سلولی و مسیرهای اتصال بین سلول‌ها به‌صورت تحلیلی پیاده شد. گرادیان‌گیری و به‌روزرسانی پارامتر جدا هستند، بنابراین هیچ وزن در میانهٔ محاسبهٔ یک گرادیان تغییر نمی‌کند. بررسی تفاضل محدود چندین کلاس یال تحریکی و مهاری، خروجی و bias را پوشش می‌دهد؛ اگر حساسیت بازگشتی عملاً صفر باشد، نتیجه inconclusive است و خطای ساختگی صفر گزارش نمی‌شود.')
heading('یادگیری محلی با ردّ صلاحیت',2)
para('الگوریتم جدید واقعاً رو به جلو حرکت می‌کند و برای هر سیناپس یک ردّ Kبعدی نگه می‌دارد. مشتق بازخورد beta درون همان سلول حفظ می‌شود، اما مسیر مشتق از فعالیت دیگر نورون‌ها در eligibility محلی حذف می‌شود. سیگنال آموزش یک تصویر ثابت و تصادفی از خطای جاری خروجی است. پس این یک تقریب مشتق شبکه است؛ برابری دقیق با BPTT یا اجرای زیستی واقعی e-prop ادعا نمی‌شود [۵،۶].')
ltr('E_(ij,t) = (1-ds_i) E_(ij,t-1) + a_i scale_i sign_j z_(j,t-1)',9)
ltr('             + beta_i gprime_(i,t-1) mean_k(E_(ij,t-1))',9)
para('بازخورد تصادفی فقط به نورون‌های تحریکی محدود نشده است؛ همهٔ نوع‌های پس‌سیناپسی می‌توانند سیگنال آموزش داشته باشند. تفاوت نسخهٔ قبل این بود که BPTT می‌توانست اتصالات ورودی نورون‌های مهاری را آموزش دهد اما روش محلی بازخورد صفر به آن‌ها داشت. اکنون مجموعهٔ پارامترهای بالقوهٔ قابل‌آموزش یکسان است.')
heading('کنترل بدون beta و reservoir',2)
para('در کنترل بدون beta فقط جملهٔ مشتق beta حذف می‌شود. تمام K کانال، دینامیک forward و سایر تنظیمات یکسان می‌مانند؛ این روش RFLO نامیده نمی‌شود چون حذف هم‌زمان فیلترها با یک مقایسهٔ معتبر RFLO یکی نیست. در reservoir همهٔ اتصالات بازگشتی دقیقاً ثابت‌اند و فقط readout یاد می‌گیرد؛ حتی weight decay یا سقف optimizer هم نباید به اتصالات ثابت آن دست بزند.')
heading('مدولاسیون اسکالر و محدودیت زیستی',2)
para('روش node perturbation از یک برآوردگر likelihood-ratio برای جریان نویزی گاوسی استفاده می‌کند: تاریخچهٔ اغتشاش و فعالیت پیش‌سیناپسی در یک eligibility جمع و با مزیتِ زیان هر trial ضرب می‌شود. خط پایه از اجرای تمیز همان ورودی ساخته می‌شود و از اغتشاشِ اجرای نویزی مستقل است. هزینهٔ اجرای اضافی ثبت می‌شود. در نویز غیرصفر هدف این برآوردگر امید زیان شبکهٔ نویزی است؛ به همین دلیل BPTT با نویز متناظر نیز در کنترل‌ها وجود دارد [۷].')
para('سیگنال اسکالر در این روش مربوط به به‌روزرسانی اتصال‌های بازگشتی است. readout همچنان با قاعدهٔ دلتا و برچسب صحیح آموزش می‌بیند؛ بنابراین کل سامانه یک عاملِ صرفاً پاداش‌محور و بدون آموزش نظارت‌شده معرفی نمی‌شود.')
para('نزدیک‌بودن سازوکار به قاعدهٔ سه‌عاملی به معنای اثبات وجود دقیق آن در مغز نیست. دسترسی به مشتق تابع سلول، سیگنال خطای آموزشی یا اغتشاش تزریق‌شده و به‌روزرسانی در انتهای minibatch همگی فرض‌اند. Adam و clipping سراسری نیز اجزای زیستی محسوب نمی‌شوند؛ پیکربندی مستقل SGD بدون این دو تسهیل ارائه شده است. این تفکیک اجازه می‌دهد ادعای محاسباتی دقیق بماند.')

heading('۸. تحلیل بعد از آموزش و کنترل نشت')
para('در تحلیل قدیمی، محور choice و مقدار d-prime روی همان trialها ساخته و ارزیابی می‌شدند. این کار می‌تواند تفکیک را حتی با نویز خوش‌بینانه کند [۸]. اکنون PCA، مقیاس‌گذاری، محورهای choice/context و ضرایب decoder فقط روی analysis-fit یاد گرفته می‌شوند و روی analysis-eval مستقل ارزیابی می‌شوند. d-prime علامت‌دار است و مقدار منفی یا نامعین حفظ می‌شود؛ بعد مؤثرِ جمعیت کاملاً ساکت نیز عدد موفق ساختگی نمی‌گیرد.')
para('هندسه در stimulus، delay و decision گزارش می‌شود. مسیر میانگین هر شرط در مختصات PCهای fit همراه تعداد trialهای موجود در هر زمان ذخیره می‌شود. decoding زمانی، مراحل نسبی تأخیر با طول متفاوت را مقایسه می‌کند. checkpointهای انتخاب‌شده روی یک بانک تحلیل مشترک بررسی می‌شوند تا تفاوت نمودار صرفاً تفاوت نمونهٔ ارزیابی نباشد.')
para('برای ارتباط نورون با نقش مدار، ویژگی‌های برازش و متادیتای فیزیولوژی همراه فعالیت تأخیر، انتخاب‌پذیری، درجهٔ گراف و وزن readout ذخیره می‌شوند. برای tupleهای مصنوعی، برچسب روشن وجود دارد. همبستگی در سطح هر شبکه محاسبه و سپس روی شبکه‌ها خلاصه می‌شود؛ ۴۸۰ نورون از چند شبکه ۴۸۰ تکرار مستقل زیستی نیستند [۹]. این آمار اکتشافی و مشروط به همین بانک سلولی است و به‌خودی‌خود اثر علی فیزیولوژی را ثابت نمی‌کند.')
para('alignment گرادیان محلی با مشتق کامل و پروفایل زمانی اندازهٔ گرادیان روی trialهای تشخیصی مستقل ثبت می‌شود. این تحلیل می‌تواند نشان دهد تقریب در چه شرایطی با جهت BPTT همسو یا ناهمسو است، اما همسویی عددی به‌تنهایی مکانیسم یادگیری زیستی یا کافی بودن آن برای رفتار را اثبات نمی‌کند.')

heading('۹. مداخله و دینامیک')
para('خاموشی نوع سلول اکنون با نام PV، SST یا VIP، در phase مشخص و دوز مشخص اعمال می‌شود. کنترل sham و کنترل هم‌اندازه وجود دارد و ورودی و نویز هر مقایسه یکسان‌اند. معیارها هم در سطح میانگین شبکه و هم در سطح trial ذخیره می‌شوند. این مداخله clamp ایده‌آل نرخ است و مدل نور، opsin، تغییر غشاء یا اثرهای فیزیولوژیک واقعی اپتوژنتیک نیست.')
para('اغتشاش جریان با واحد pA به مقیاس داخلی صحیح تبدیل می‌شود. فاصلهٔ نرخ‌ها و فاصلهٔ کل حالت Kبعدی با هم گزارش می‌شوند؛ منفی بودن recovery یا افزایش فاصله سانسور نمی‌شود. برای تأخیر یک bin که زمان جداگانه‌ای برای اغتشاش و بهبود ندارد، assay صریحاً غیرقابل‌اعمال علامت می‌خورد.')
para('برای بررسی واقعی‌تر جاذب، ابزار جداگانهٔ حل نقطهٔ ثابت و ژاکوبین کامل NK×NK اضافه شد [۱۰]. حل‌گر باید هم موفقیت عددی و هم باقیماندهٔ کوچک را نشان دهد؛ ناموفق بودن حل یا قرارگرفتن نزدیک شکست مشتق ReLU پنهان نمی‌شود. حتی یک نقطهٔ ثابت با شعاع طیفی کمتر از یک فقط پایداری موضعی تحت ورودی ثابت را پشتیبانی می‌کند؛ یکتایی، حوضهٔ جذب بزرگ و نقش حافظه‌ای از آن نتیجه نمی‌شود.')

heading('۱۰. آمار، بازتولید و هزینهٔ محاسبه')
para('واحد تکرار آماری، اجرای مستقل شبکه با نمونه‌گیری سلول و وزن اولیهٔ مستقل است. وضعیت‌ها با همان seed جفت‌اند. bootstrap کل شبکه‌ها را بازنمونه‌گیری می‌کند و آزمون sign-flip روی اختلاف جفت‌ها انجام می‌شود. اگر شبکه‌ای در یک شرط نباشد، pairing ناقص با حذف خاموش برطرف نمی‌شود. خانوادهٔ اصلی هفت اثر فاکتوریل روی دقت trialهای متعارض در تأخیر جدید است و تصحیح Holm روی همان خانواده اعمال می‌شود.')
para('با سه جفت، کوچک‌ترین p-value دوطرفهٔ آزمون دقیق برابر ۰٫۲۵ است. بنابراین این پایلوت به‌تنهایی نمی‌تواند برتری معنی‌دارِ ۰٫۰۵ ارائه کند، حتی اگر همهٔ جهت‌های اختلاف یکسان باشند. فاصله‌های bootstrap با تعداد بسیار کم شبکه پوشش مطمئن تلقی نمی‌شوند. تعداد ۱۲ seed در قالب تأییدی یک نقطهٔ شروع برنامه‌ریزی است، نه محاسبهٔ توان آماری انجام‌شده.')
para('هر draw تصادفی از یک namespace پایدار و شمارهٔ گام ساخته می‌شود. افزودن ارزیابی، ترتیب اجرای شرایط یا ادامه از checkpoint مثال‌های آموزشی را تغییر نمی‌دهد. انتخاب checkpoint فقط با BCE اعتبارسنجی است و تمام گام‌های بودجه اجرا می‌شوند. tuner تعداد گزینه، seed، گام و batch یکسان به روش‌ها می‌دهد و مسیر test را فراخوانی نمی‌کند. انتخاب تنظیمات شبکهٔ ۳۲نورونی جانشین tuning در معماری نهایی ۱۰۰نورونی نیست.')
para('checkpointها NPZ امن شامل وزن، وضعیت optimizer و متادیتای JSON هستند و pickle لازم ندارند. هش کد، پیکربندی، فایل مدل و متادیتا، cohort سلولی و ساختار شبکه کنترل می‌شوند. بارگذاری ناقص وزن‌ها قبل از تأیید کامل انجام نمی‌شود. آزمون resume، ادامهٔ قطع‌شده و اجرای پیوسته را تا آرایه‌های وزن و momentهای optimizer مقایسه می‌کند.')
para('بودجهٔ نمونه و بودجهٔ محاسبه یک مفهوم نیستند. روش perturbation با baseline تمیز دو rollout دارد و local eligibility حافظهٔ وابسته به تعداد یال‌ها و K نگه می‌دارد. تعداد trial، bin شبیه‌سازی‌شده، زمان آموزش و وضعیت clipping/projection در خروجی ثبت می‌شوند. برای ادعای برتری کارایی، مقایسهٔ جداگانه با محاسبهٔ هم‌بودجه و سخت‌افزار مشخص لازم است.')

heading('۱۱. نتایج واقعاً اجراشده')
para('پایلوت فاکتوریل با ۳۲ نورون، ۳ بذر، ۳۰۰ به‌روزرسانی minibatch و batch برابر ۳۲ اجرا شد. برای هر شبکه ۱۲۸ trial اعتبارسنجی، ۲۵۶ trial در هر یک از پنج آزمون نهایی و ۶۴ trial در هر بانک تحلیل وجود دارد. تنظیمات این پایلوت از قبل در فایل اجرا مشخص است؛ این آزمایش توسعه‌ای بوده و نتیجهٔ تأییدی بانک رزروشده نیست.')
para('در جدول، میانگین و انحراف معیار بین سه شبکه بر حسب درصد نوشته شده است. ID آزمون هم‌توزیع، OOD delay تأخیرِ جدید و conflict زیرمجموعهٔ شواهد متعارض همان آزمون است. جمع‌زدن trialها و محاسبهٔ خطای استاندارد به‌عنوان جایگزین تکرار مستقل شبکه انجام نشده است.')
labels={
 'authentic__type_prior__bptt':'Intact / type prior / BPTT',
 'authentic__type_prior__local':'Intact / type prior / local',
 'authentic__degree_rewired__bptt':'Intact / rewired / BPTT',
 'authentic__degree_rewired__local':'Intact / rewired / local',
 'within_type_shuffle__type_prior__bptt':'Shuffled / type prior / BPTT',
 'within_type_shuffle__type_prior__local':'Shuffled / type prior / local',
 'within_type_shuffle__degree_rewired__bptt':'Shuffled / rewired / BPTT',
 'within_type_shuffle__degree_rewired__local':'Shuffled / rewired / local',
 'local_no_beta':'Local without beta derivative','readout_only':'Frozen reservoir',
 'type_homogeneous_local':'Type homogeneous / local','fast_decay_local':'Fast static-matched / local',
 'cross_type_local':'Cross-type permutation / local','node_perturbation':'Node perturbation',
 'bptt_noise_matched':'BPTT / matched training noise'}
rows=[]
for name in METRICS.condition.unique():
 vals=[]
 for split,metric in [('test_id','accuracy'),('test_delay','accuracy'),('test_delay','accuracy_incongruent')]:
  a=METRICS[(METRICS.condition==name)&(METRICS.split==split)][metric].to_numpy()*100
  vals.append(f'{a.mean():.1f} ± {a.std(ddof=1):.1f}')
 rows.append([labels.get(name,name),*vals])
table(['Condition','ID (%)','OOD delay (%)','OOD conflict (%)'],rows,[8,2.6,2.9,3.1])
para('این جدول برای تشخیص اندازهٔ اثر، ناهمگنی شبکه‌ها و مسائل باقی‌مانده مفید است. افت روی trialهای متعارض یا روی تأخیر جدید باید در متن مقاله دیده شود؛ دقت کلی بالاتر، جایگزین این آزمون‌ها نیست. خروجی هر seed، هر trial و تمام شرایط نامطلوب در بسته موجود است.')
para('یک محدودیت مهم در خود نتایج دیده می‌شود: در آزمون تأخیر جدید، نرخ میانگین فعالیتِ مدل‌های shuffle حدود ۲۶ تا ۲۷ Hz و در مدل‌های intact حدود ۱۳ تا ۱۴ Hz است؛ اشباع نیز در shuffle بیشتر است. این اندازه‌گیری میانگین روی کل trialهای آزمون است. پس اثر shuffle با تغییر رژیم کاری همراه است و نباید به‌تنهایی شاهد خالصِ سودمندی covariance فیزیولوژیک تلقی شود. حساسیت با نقطهٔ کاری و نرخ‌های کنترل‌شده برای آن ادعا لازم است.')
if (ROOT/'results/pilot/tuning/tuning_summary.json').exists():
 tunes=json.loads((ROOT/'results/pilot/tuning/tuning_summary.json').read_text())
 selections=json.loads((ROOT/'results/pilot/tuning/selection_record.json').read_text())
 para(f'علاوه بر فاکتوریل، جست‌وجوی validation-only برای چهار روش اصلی اجرا شد: {len(tunes)} ترکیب روش/گزینه و سه seed برای هر ترکیب، معادل {len(tunes)*3} اجرای توسعه‌ای. هیچ test در این جست‌وجو ارزیابی نشد. تنظیمات انتخاب‌شده در selection_record.json ذخیره شده‌اند و به‌طور پس‌نگر جای نتایج پایلوت اولیه را نگرفته‌اند.')
 table(['Learner','Selected lr_rec','Selected lr_out'],[[k,v['lr_rec'],v['lr_out']] for k,v in selections['selected_learning_rates'].items()],[7,4,4])
if (ROOT/'results/pilot/tuned/metrics.csv').exists():
 tuned=pd.read_csv(ROOT/'results/pilot/tuned/metrics.csv')
 para('پس از انتخاب نرخ‌ها، چهار روش اصلی روی بذرهای توسعه‌ای تازهٔ ۳۰۳، ۴۰۴ و ۵۰۵ اجرا شدند؛ ۱۲ اجرای جدید از مجموعهٔ رزروشدهٔ تأییدی استفاده نکردند. جدول زیر این اجرای پس از انتخاب را نشان می‌دهد. تفاوت seed و تنظیمات باعث می‌شود این جدول با جدول قبلی در یک آزمون آماری ادغام نشود.')
 tr=[]
 for name in tuned.condition.unique():
  vals=[]
  for split,metric in [('test_id','accuracy'),('test_delay','accuracy'),('test_delay','accuracy_incongruent')]:
   a=tuned[(tuned.condition==name)&(tuned.split==split)][metric].to_numpy()*100
   vals.append(f'{a.mean():.1f} ± {a.std(ddof=1):.1f}')
  tr.append([name,*vals])
 table(['Selected-setting method','ID (%)','OOD delay (%)','OOD conflict (%)'],tr,[7,3,3,3.6])
fd=V['finite_difference_actual_allen']
para(f'در ممیزی عددی انتشار، {fd["checked_recurrent_edges"]} اتصال بازگشتی بررسی و {fd["nonzero_recurrent_checks"]} اتصال دارای حساسیت غیرصفر تشخیص داده شد. خطای نسبی L2 برابر {fd["relative_l2_error"]:.3e} و بیشینهٔ خطای مطلق {fd["max_absolute_error"]:.3e} بود. مختصات اتصال‌ها و مشتق‌ها در validation.json ثبت شده‌اند. آزمون‌ها برابری forward مستقل، عدم تغییر فیزیولوژی و مشتق محلی تحت تاریخچهٔ پیش‌سیناپسی ثابت را نیز بررسی می‌کنند. این موفقیت دربارهٔ صحت پیاده‌سازی است، نه دربارهٔ موفقیت مقاله.')

heading('۱۲. مسیر قابل‌دفاع برای مقاله')
para('پرسش اصلی پیشنهادی این است: آیا همراهی طبیعی پارامترهای تاریخچه و فعال‌سازیِ هر سلول، در حضور prior اتصال مبتنی بر نوع، به یادگیری محلی و تعمیم حافظه کمک می‌کند و این اثر با BPTT چه تفاوتی دارد؟ پاسخ باید به کنترل shuffle، کنترل گراف، اثر متقابل و عملکرد trialهای متعارض متکی باشد. صرف استفاده از سلول‌های Allen و چند نمودار PCA برای ادعای نوآوری کافی نیست.')
para('مرحلهٔ نخست، نهایی کردن تنظیمات و بازهٔ هزینه در معماری هدف با توسعهٔ جداگانه است. سپس فرضیه‌ها، endpoint اصلی، خانوادهٔ آزمون‌ها، تعداد seed بر اساس دقت برآورد مطلوب، معیار شکست اجرا و همهٔ حساسیت‌ها باید پیش از مشاهدهٔ مجموعهٔ تأییدی ثابت شوند. قالب ۹۶اجرایی موجود، هشت وضعیت فاکتوریل با ۱۲ seed، ۱۰۰ نورون و ۱۵۰۰ گام است؛ این قالب هنوز اجرا و به‌صورت بیرونی preregister نشده است.')
para('مرحلهٔ بعد، تکرار در task با context دیرهنگام، تغییر مقیاس جریان، دامنهٔ اتصال، بازخورد تصادفی، شدت نویز و قیود به‌روزرسانی است. در صورت ادعای مکانیسم سلولی باید آثار نقطهٔ کاری، اختلاف gain غیرخطی و تعمیم به cohort مستقل بررسی شوند. برای ادعای وفاداری patch clamp، sweepهای خامِ کنارگذاشته‌شده و پروتکل‌های خارج از برازش لازم‌اند. این بسته آن دادهٔ خام را در اختیار ندارد.')
para('مداخلهٔ سلولی باید نسبت به کنترل هم‌اندازه و افت نرخ عمومی تفسیر شود. همبستگی بیوفیزیک با نقش شبکه باید یا در چارچوب شبکهٔ مستقل و این بانک محدود گزارش شود، یا با مدل مناسب برای وابستگی specimen و donor توسعه یابد. ادعای جاذب حافظه نیازمند چند نقطهٔ شروع وابسته به حالت، همگرایی، پایداری کل حالت و شواهد حفظ اطلاعات رفتاری است؛ recovery یک پالس کافی نیست.')
para('اگر reservoir، شبکهٔ shuffle یا توپولوژی تصادفی هم‌تراز یا بهتر باشد، آن نتیجه باید حفظ شود. چنین نتیجه‌ای می‌تواند فرضیه را محدود کند یا نشان دهد task هنوز برای آشکارکردن اثر موردنظر مناسب نیست. بازطراحی حاضر برای برنده کردن از پیش تعیین‌شدهٔ حالت زیستی تنظیم نشده است. آمادگی برای ارسال به ژورنال زمانی قابل ارزیابی است که این مراحل تجربی و جایگاه نوآوری نسبت به ادبیات تکمیل شوند.')

heading('۱۳. راهنمای فایل‌ها و اجرا')
para('README.md نقطهٔ شروع است. پوشهٔ grnn_bio پیاده‌سازی اجرایی، configs طرح‌ها، tests آزمون‌ها، provenance منبع و هش‌ها، results/pilot دادهٔ اجرای واقعی، و docs توضیحات فنی را نگه می‌دارد. فایل‌های LEARNING_RULES، ANALYSIS_METHODS، STATE_DYNAMICS و RESEARCH_PROTOCOL جزئیات ریاضی و محدودیت‌ها را تکمیل می‌کنند. docs/MIGRATION.md نگاشت هر ماژول قدیمی به جایگزین آن را ارائه می‌دهد.')
ltr('python -m pip install -r requirements.txt',9)
ltr('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio validate',8)
ltr('python -m grnn_bio plan --config configs/factorial_pilot.json',8)
ltr('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \\\n  --config configs/smoke.json --output results/my_smoke',8)
para('برای ادامهٔ اجرا، همان فرمان با --resume استفاده می‌شود؛ تغییر کد، داده یا config باید با مسیر خروجی جدید همراه باشد. یک بار اجرا کردن validate و smoke، جای اجرای طرح تأییدی را نمی‌گیرد. نسخه‌های دقیق کتابخانه‌های استفاده‌شده در requirements-tested.txt ثبت شده‌اند. NumPy مبنای اجرای جدید است و اجرای GPU یا PyTorch برای نتایج تحویلی ادعا نشده است.')
para('مجوز نرم‌افزار Allen همان متن اصلی با بند غیرتجاری است و در provenance/upstream/LICENSE.txt حفظ شده است. اظهار مجوز داده در metadata منبع جدا ثبت شده و این بسته کد یا دادهٔ ثالث را به‌طور ضمنی MIT یا BSD بدون محدودیت معرفی نمی‌کند. در انتشار عمومی، انتساب و مجوز سازگار باید همراه پروژه بمانند.')

heading('۱۴. منابع و ردیابی شواهد')
refs=[
 ('1','Yeung C, Ganjam K, Zhang Z, Mihalas S, Lu Z. A dataset of differentiable biologically derived single-neuron models. Cell Reports Methods, 2026.','https://doi.org/10.1016/j.crmeth.2026.101638'),
 ('2','Pfeffer CK et al. Inhibition of inhibition in visual cortex. Nature Neuroscience, 2013.','https://doi.org/10.1038/nn.3446'),
 ('3','Pi HJ et al. Cortical interneurons that specialize in disinhibitory control. Nature, 2013.','https://doi.org/10.1038/nature12676'),
 ('4','Mante V et al. Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 2013.','https://doi.org/10.1038/nature12742'),
 ('5','Bellec G et al. A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications, 2020.','https://doi.org/10.1038/s41467-020-17236-y'),
 ('6','Murray JM. Local online learning in recurrent networks with random feedback. eLife, 2019.','https://doi.org/10.7554/eLife.43299'),
 ('7','Miconi T. Biologically plausible learning in recurrent neural networks reproduces neural dynamics observed during cognitive tasks. eLife, 2017.','https://doi.org/10.7554/eLife.20899'),
 ('8','Kriegeskorte N et al. Circular analysis in systems neuroscience: the dangers of double dipping. Nature Neuroscience, 2009.','https://doi.org/10.1038/nn.2303'),
 ('9','Aarts E et al. A solution to dependency: using multilevel analysis to accommodate nested data. Nature Neuroscience, 2014.','https://doi.org/10.1038/nn.3648'),
 ('10','Sussillo D, Barak O. Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. Neural Computation, 2013.','https://pubmed.ncbi.nlm.nih.gov/23272922/'),
]
for num,title,url in refs:
 ltr(f'[{num}] {title}',9);ltr(url,8)
para('شواهد مربوط به خطاهای نسخهٔ ۶ از فایل‌های خود پروژه و ممیزی ذخیره‌شده استخراج شده‌اند. نتایج عددی جدید از CSV، JSON و checkpointهای همین نسخه استخراج می‌شوند. مقاله‌ها برای مبنای مدل، قواعد یادگیری و روش‌شناسی ارجاع داده شده‌اند؛ هیچ‌کدام مقادیر دقیقِ prior اتصال یا نتیجهٔ اجرای جدید این پروژه را تأیید تجربی نمی‌کنند.')

# Footer with a real Word PAGE field; no decorative header or horizontal rules.
p=section.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
section.footer_distance=Cm(0.8)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
field_run=OxmlElement('w:r');field_text=OxmlElement('w:t');field_text.text='1'
field_run.append(field_text);field.append(field_run);p._p.append(field)
DOC.core_properties.title='GRNN GFR version 7: Persian change and validation report'
DOC.core_properties.subject='Methodological audit, implemented changes, actual pilot results and publication protocol'
DOC.core_properties.author='Research project report'
OUTPUT.parent.mkdir(parents=True,exist_ok=True);DOC.save(OUTPUT)
print(OUTPUT)
