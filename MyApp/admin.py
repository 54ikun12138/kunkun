from __future__ import annotations

from django.contrib import admin

from .models import ConsumptionRecord, CustomUser


@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ("username", "display_name", "email", "is_active", "is_staff")
    search_fields = ("username", "display_name", "email")
    list_filter = ("is_staff", "is_active", "is_data_viewer", "is_data_editor")
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        ("个人信息", {"fields": ("display_name", "avatar", "email", "phone", "bio")}),
        (
            "权限",
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "is_data_viewer",
                    "is_data_editor",
                    "groups",
                    "user_permissions",
                )
            },
        ),
        ("重要时间", {"fields": ("last_login", "date_joined")}),
    )
    readonly_fields = ("last_login", "date_joined")


@admin.register(ConsumptionRecord)
class ConsumptionRecordAdmin(admin.ModelAdmin):
    list_display = (
        "transaction_serial",
        "student_no",
        "merchant_name",
        "amount",
        "transaction_time",
    )
    search_fields = ("transaction_serial", "student_no", "merchant_name")
    list_filter = ("merchant_name", "settle_date", "flow_flag")
    autocomplete_fields = ("user",)

