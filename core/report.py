import io
import os
from fpdf import FPDF
from datetime import datetime


def _find_chinese_font():
    candidates = [
        # Windows
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simsun.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simkai.ttf',
        # macOS
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/Supplemental/Songti.ttc',
        # Linux
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = _find_chinese_font()
        if font_path:
            self.add_font('Chinese', '', font_path)
            self.font_name = 'Chinese'
        else:
            self.font_name = 'Helvetica'

    def header(self):
        self.set_font(self.font_name, '', 10)
        self.cell(0, 8, 'Software Project Effort Estimation Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_name, '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font(self.font_name, '', 14)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font(self.font_name, '', 10)
        self.set_fill_color(76, 114, 176)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, str(h), border=1, fill=True, align='C')
        self.ln()
        self.set_text_color(0, 0, 0)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 8, str(val), border=1, align='C')
            self.ln()
        self.ln(4)


def generate_pdf(project_data: dict, predictions: dict, recommended_model: str):
    pdf = ReportPDF()
    pdf.add_page()

    pdf.set_font(pdf.font_name, '', 18)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    pdf.cell(0, 12, f"Project: {project_data['project_name']}", new_x='LMARGIN', new_y='NEXT')
    pdf.set_font(pdf.font_name, '', 10)
    pdf.cell(0, 8, f"Generated: {now}", new_x='LMARGIN', new_y='NEXT')
    pdf.ln(6)

    pdf.section_title('1. Project Parameters')
    param_labels = {
        'function_points':    'Function Points (功能点计数)',
        'project_complexity': 'Project Complexity (项目复杂度)',
        'code_size_kloc':     'Code Size KLoC (代码规模/千行)',
        'team_experience':    'Team Experience (团队经验)',
        'tool_maturity':      'Tool Maturity (工具成熟度)',
        'dev_mode':           'Development Mode (开发模式)',
        'language_type':      'Language Type (语言类型)',
    }
    dev_mode_names = {0: 'Organic', 1: 'Semi-detached', 2: 'Embedded'}
    lang_type_names = {0: 'Low-level', 1: 'High-level', 2: 'Very high-level'}
    param_rows = []
    for key, label in param_labels.items():
        val = project_data[key]
        if key == 'dev_mode':
            val = f"{val} ({dev_mode_names.get(val, '')})"
        elif key == 'language_type':
            val = f"{val} ({lang_type_names.get(val, '')})"
        param_rows.append([label, str(val)])
    pdf.add_table(['Parameter', 'Value'], param_rows, [100, 90])

    pdf.section_title('2. Model Predictions')
    pred_rows = []
    for model_name, value in predictions.items():
        marker = ' *' if model_name == recommended_model else ''
        pred_rows.append([model_name + marker, f"{value:.2f}"])
    pdf.add_table(['Model', 'Predicted Effort (Person-Months)'], pred_rows, [110, 80])
    pdf.set_font(pdf.font_name, '', 9)
    pdf.cell(0, 6, f"* Recommended model: {recommended_model}", new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    rec_effort = predictions[recommended_model]
    pdf.section_title('3. Schedule Estimation')
    schedule_rows = []
    for team_size in [1, 2, 3, 5]:
        months = rec_effort / team_size
        schedule_rows.append([str(team_size), f"{months:.1f}"])
    pdf.add_table(['Team Size (persons)', 'Estimated Duration (months)'], schedule_rows, [95, 95])

    pdf.section_title('4. Conclusion')
    best_team = 3 if rec_effort > 6 else 2 if rec_effort > 3 else 1
    duration = rec_effort / best_team
    pdf.set_font(pdf.font_name, '', 11)
    conclusion = (
        f"Based on {recommended_model}, the estimated effort for project "
        f"'{project_data['project_name']}' is {rec_effort:.2f} person-months. "
        f"Recommended team size: {best_team} persons, "
        f"estimated duration: {duration:.1f} months."
    )
    pdf.multi_cell(0, 7, conclusion)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.getvalue()







