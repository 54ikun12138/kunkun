from __future__ import annotations

import io

from django import forms
from django.contrib.auth.forms import UserCreationForm

from .models import CustomUser


class LoginForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=150)
    password = forms.CharField(label="密码", widget=forms.PasswordInput)


class RegistrationForm(UserCreationForm):
    display_name = forms.CharField(label="显示名", max_length=64, required=False)
    email = forms.EmailField(label="邮箱")
    phone = forms.CharField(label="联系电话", max_length=32, required=False)

    class Meta(UserCreationForm.Meta):
        model = CustomUser
        fields = ("username", "display_name", "email", "phone")


class CsvUploadForm(forms.Form):
    file = forms.FileField(
        label="消费数据 CSV 文件",
        help_text="请选择包含学工号等字段的 CSV 文件，系统将自动剔除教师记录。",
    )

    def clean_file(self) -> io.BytesIO:
        uploaded = self.cleaned_data["file"]
        if not uploaded.name.lower().endswith(".csv"):
            raise forms.ValidationError("目前仅支持 CSV 格式文件。")
        return uploaded


class ProfileForm(forms.ModelForm):
    """个人信息编辑表单"""

    class Meta:
        model = CustomUser
        fields = ("display_name", "email", "phone", "bio", "avatar", "first_name", "last_name")
        labels = {
            "display_name": "昵称/显示名",
            "email": "邮箱",
            "phone": "联系电话",
            "bio": "个人简介",
            "avatar": "头像",
            "first_name": "名",
            "last_name": "姓",
        }
        widgets = {
            "bio": forms.Textarea(attrs={"rows": 3}),
        }

