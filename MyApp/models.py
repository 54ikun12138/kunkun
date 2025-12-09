from __future__ import annotations

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    """系统管理用户，主要用于登录与后台操作。"""

    display_name = models.CharField("昵称/显示名", max_length=64, blank=True)
    avatar = models.ImageField("头像", upload_to="avatars/", blank=True, null=True)
    phone = models.CharField("联系电话", max_length=32, blank=True)
    bio = models.CharField("个人简介", max_length=255, blank=True)
    is_data_viewer = models.BooleanField("可查看数据", default=True)
    is_data_editor = models.BooleanField("可编辑数据", default=False)

    class Meta:
        verbose_name = "用户"
        verbose_name_plural = "用户"

    def __str__(self) -> str:
        return self.display_name or self.get_full_name() or self.username


class ConsumptionRecord(models.Model):
    """参考 9 月份消费数据，记录每笔交易。"""

    student_no = models.CharField("学工号", max_length=32, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name="consumptions",
        verbose_name="关联用户",
    )
    transaction_serial = models.CharField("流水号", max_length=32, unique=True)
    merchant_name = models.CharField("商户名称", max_length=128)
    transaction_event = models.CharField("交易事件", max_length=64)
    amount = models.DecimalField("交易金额(元)", max_digits=10, decimal_places=2)
    transaction_time = models.DateTimeField("交易时间", db_index=True)
    settle_date = models.DateField("入账日期")
    pos_code = models.CharField("POS代码", max_length=32)
    flow_flag = models.CharField("流水标志", max_length=32, default="正常流水")
    created_at = models.DateTimeField("记录创建时间", auto_now_add=True)
    updated_at = models.DateTimeField("记录更新时间", auto_now=True)

    class Meta:
        verbose_name = "消费流水"
        verbose_name_plural = "消费流水"
        indexes = [
            models.Index(fields=["settle_date"], name="idx_settle_date"),
            models.Index(fields=["merchant_name"], name="idx_merchant"),
        ]
        ordering = ["-transaction_time"]

    def __str__(self) -> str:
        return f"{self.student_no}-{self.transaction_serial}"

