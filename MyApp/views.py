from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

import numpy as np
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Avg, Count, Max, Min, Q, Sum
from django.db.models.functions import ExtractHour, ExtractWeekDay, TruncDate
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .forms import CsvUploadForm, LoginForm, ProfileForm, RegistrationForm
from .models import ConsumptionRecord


def login_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("MyApp:dashboard")

    form = LoginForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = authenticate(
            request,
            username=form.cleaned_data["username"],
            password=form.cleaned_data["password"],
        )
        if user is not None:
            login(request, user)
            if ConsumptionRecord.objects.count() == 0:
                messages.warning(request, "系统暂无消费数据，请先上传 CSV 文件。")
                return redirect("MyApp:upload_csv")
            messages.success(request, "登录成功，欢迎回来！")
            return redirect("MyApp:dashboard")
        messages.error(request, "用户名或密码错误。")

    return render(
        request,
        "auth/login.html",
        {"form": form, "data_count": ConsumptionRecord.objects.count()},
    )


def register_view(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("MyApp:dashboard")

    form = RegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "注册成功，已自动登录。")
        return redirect("MyApp:dashboard")

    return render(request, "auth/register.html", {"form": form})


def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    messages.info(request, "您已退出系统。")
    return redirect("MyApp:login")


@login_required
def dashboard(request: HttpRequest) -> HttpResponse:
    consumption_count = ConsumptionRecord.objects.count()
    total_amount = (
        ConsumptionRecord.objects.aggregate(total=Sum("amount"))["total"] or Decimal("0")
    )
    student_count = (
        ConsumptionRecord.objects.values("student_no").distinct().count()
        if consumption_count
        else 0
    )
    merchant_count = (
        ConsumptionRecord.objects.values("merchant_name").distinct().count()
        if consumption_count
        else 0
    )

    avg_ticket = (
        (total_amount / Decimal(consumption_count)).quantize(Decimal("0.01"))
        if consumption_count
        else Decimal("0.00")
    )

    latest_time = (
        ConsumptionRecord.objects.order_by("-transaction_time").values_list(
            "transaction_time", flat=True
        )
    ).first()
    if latest_time is None:
        trend_since = timezone.now() - timedelta(days=30)
    else:
        if timezone.is_naive(latest_time):
            latest_time = timezone.make_aware(
                latest_time, timezone.get_current_timezone()
            )
        trend_since = latest_time - timedelta(days=30)
    base_qs = ConsumptionRecord.objects.filter(transaction_time__gte=trend_since)
    trend_qs = (
        base_qs.annotate(day=TruncDate("transaction_time"))
        .values("day")
        .annotate(amount=Sum("amount"), count=Count("id"))
        .order_by("day")
    )
    trend_labels: list[str] = []
    trend_amounts: list[float] = []
    trend_counts: list[int] = []
    for entry in trend_qs:
        day = entry.get("day")
        if day is None:
            continue
        label = day.strftime("%m-%d")
        trend_labels.append(label)
        trend_amounts.append(float(entry.get("amount") or 0))
        trend_counts.append(entry.get("count") or 0)

    event_stats = list(
        ConsumptionRecord.objects.values("transaction_event")
        .annotate(count=Count("id"))
        .order_by("-count")[:5]
    )
    event_chart_available = len(event_stats) > 1 and any(
        item["count"] for item in event_stats
    )
    merchant_stats = list(
        ConsumptionRecord.objects.values("merchant_name")
        .annotate(amount=Sum("amount"))
        .order_by("-amount")[:8]
    )

    category_totals: dict[str, dict[str, Decimal | int]] = defaultdict(
        lambda: {"amount": Decimal("0.00"), "count": 0}
    )
    for entry in ConsumptionRecord.objects.values("merchant_name").annotate(
        amount=Sum("amount"), count=Count("id")
    ):
        category = categorize_merchant(entry["merchant_name"])
        category_totals[category]["amount"] += entry["amount"] or Decimal("0.00")
        category_totals[category]["count"] += entry["count"]
    category_stats = [
        {"name": name, "amount": float(value["amount"]), "count": value["count"]}
        for name, value in category_totals.items()
    ]
    category_stats.sort(key=lambda item: item["amount"], reverse=True)

    pos_code_stats = list(
        ConsumptionRecord.objects.values("pos_code")
        .annotate(count=Count("id"), amount=Sum("amount"))
        .order_by("-count")[:6]
    )
    flow_flag_stats = list(
        ConsumptionRecord.objects.values("flow_flag")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    def mask_student_no(student_no):
        if len(student_no) > 8:
            return student_no[:4] + '*' * (len(student_no) - 8) + student_no[-4:]
        return student_no
    student_top = list(
        ConsumptionRecord.objects.values("student_no")
        .annotate(amount=Sum("amount"), count=Count("id"))
        .order_by("-amount")[:6]
    )
    for record in student_top:
        record["student_no"] = mask_student_no(record["student_no"])
    daily_qs = (
        base_qs.values("settle_date")
        .annotate(amount=Sum("amount"), count=Count("id"))
        .order_by("settle_date")
    )
    daily_labels = []
    daily_amounts = []
    daily_counts = []
    for entry in daily_qs:
        settle_date = entry.get("settle_date")
        if not settle_date:
            continue
        daily_labels.append(settle_date.strftime("%m-%d"))
        daily_amounts.append(float(entry.get("amount") or 0))
        daily_counts.append(entry.get("count") or 0)

    if not trend_labels and daily_labels:
        trend_labels = daily_labels
        trend_amounts = daily_amounts
        trend_counts = daily_counts

    today = timezone.now().date()
    current_period_start = today - timedelta(days=6)
    previous_period_start = current_period_start - timedelta(days=7)
    previous_period_end = current_period_start - timedelta(days=1)

    def avg_ticket_range(start: datetime.date, end: datetime.date) -> Decimal:
        qs = ConsumptionRecord.objects.filter(
            transaction_time__date__gte=start, transaction_time__date__lte=end
        )
        stats = qs.aggregate(amount=Sum("amount"), cnt=Count("id"))
        if not stats["cnt"]:
            return Decimal("0.00")
        return (stats["amount"] or Decimal("0")) / Decimal(stats["cnt"])

    avg_ticket_compare = {
        "current": avg_ticket_range(current_period_start, today).quantize(
            Decimal("0.01")
        ),
        "previous": avg_ticket_range(previous_period_start, previous_period_end).quantize(
            Decimal("0.01")
        ),
    }

    heatmap_start = today - timedelta(days=6)
    heatmap_qs = (
        ConsumptionRecord.objects.filter(transaction_time__date__gte=heatmap_start)
        .annotate(day=TruncDate("transaction_time"), hour=ExtractHour("transaction_time"))
        .values("day", "hour")
        .annotate(amount=Sum("amount"))
    )
    day_labels = [
        (heatmap_start + timedelta(days=i)).strftime("%m-%d") for i in range(7)
    ]
    heatmap_matrix = []
    day_index_map = {
        (heatmap_start + timedelta(days=i)): i for i in range(7)
    }
    for entry in heatmap_qs:
        day = entry.get("day")
        hour = entry.get("hour")
        if day not in day_index_map or hour is None:
            continue
        heatmap_matrix.append(
            [day_index_map[day], hour, float(entry.get("amount") or 0)]
        )

    latest_records = ConsumptionRecord.objects.order_by("-transaction_time")[:8]

    chart_payload = json.dumps(
        {
            "trend": {
                "labels": trend_labels,
                "amounts": trend_amounts,
                "counts": trend_counts,
            },
            "events": event_stats,
            "merchants": merchant_stats,
            "daily": {
                "labels": daily_labels,
                "amounts": daily_amounts,
            },
            "merchantBubbles": [
                {"name": entry["merchant_name"], "value": float(entry["amount"])}
                for entry in merchant_stats
            ],
            "posCodes": pos_code_stats,
            "flowFlags": flow_flag_stats,
            "categories": category_stats,
            "avgTicket": {
                "current": float(avg_ticket_compare["current"]),
                "previous": float(avg_ticket_compare["previous"]),
            },
            "students": [
                {
                    "student_no": entry["student_no"],
                    "amount": float(entry["amount"]),
                    "count": entry["count"],
                }
                for entry in student_top
            ],
            "heatmap": {
                "days": day_labels,
                "matrix": heatmap_matrix,
            },
        },
        cls=DjangoJSONEncoder,
    )

    return render(
        request,
        "dashboard.html",
        {
            "consumption_count": consumption_count,
            "total_amount": total_amount,
            "avg_ticket": avg_ticket,
            "student_count": student_count,
            "merchant_count": merchant_count,
            "latest_records": latest_records,
            "chart_payload": chart_payload,
            "event_stats": event_stats,
            "event_chart_available": event_chart_available,
            "pos_code_stats": pos_code_stats,
            "flow_flag_stats": flow_flag_stats,
            "category_stats": category_stats,
            "avg_ticket_compare": avg_ticket_compare,
            "student_top": student_top,
            "heatmap_days": day_labels,
        },
    )


@login_required
def upload_csv_view(request: HttpRequest) -> HttpResponse:
    form = CsvUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        file = form.cleaned_data["file"]
        created, updated, skipped = ingest_consumption_csv(file, request.user)
        messages.success(
            request,
            f"导入完成：新增 {created} 条，更新 {updated} 条，剔除教师数据 {skipped} 条。",
        )
        return redirect("MyApp:dashboard")

    return render(
        request,
        "uploads/upload_csv.html",
        {
            "form": form,
            "data_count": ConsumptionRecord.objects.count(),
            "upload_hint": "当前学工号 6 位数的记录会被识别为教师消费并自动剔除。",
        },
    )


def ingest_consumption_csv(file, operator) -> tuple[int, int, int]:
    """Parse uploaded CSV and sync into database."""
    file.seek(0)
    decoded = file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))
    created = updated = skipped = 0
    for row in reader:
        student_no = (row.get("学工号") or "").strip()
        if not student_no or len(student_no) == 6:
            skipped += 1
            continue

        transaction_serial = (row.get("流水号") or "").strip()
        if not transaction_serial:
            continue

        transaction_time = parse_datetime(row.get("交易时间"))
        settle_date = parse_date(row.get("入账日期"))

        defaults = {
            "student_no": student_no,
            "user": operator if operator.is_authenticated else None,
            "merchant_name": row.get("商户名称", "").strip(),
            "transaction_event": row.get("交易事件", "").strip(),
            "amount": parse_decimal(row.get("交易金额(元)")),
            "transaction_time": transaction_time,
            "settle_date": settle_date,
            "pos_code": row.get("POS代码", "").strip(),
            "flow_flag": row.get("流水标志", "").strip() or "正常流水",
        }

        _, is_created = ConsumptionRecord.objects.update_or_create(
            transaction_serial=transaction_serial,
            defaults=defaults,
        )
        if is_created:
            created += 1
        else:
            updated += 1
    return created, updated, skipped


def parse_datetime(value: str | None):
    if not value:
        return timezone.now()
    try:
        dt = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
        return timezone.make_aware(dt, timezone.get_current_timezone())
    except ValueError:
        return timezone.now()


def parse_date(value: str | None):
    if not value:
        return timezone.now().date()
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return timezone.now().date()


def parse_decimal(value: str | None) -> Decimal:
    if not value:
        return Decimal("0.00")
    try:
        return Decimal(value.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return Decimal("0.00")


def categorize_merchant(name: str | None) -> str:
    if not name:
        return "其他"
    mapping = {
        "食堂": "食堂/餐厅",
        "餐厅": "食堂/餐厅",
        "餐饮": "食堂/餐厅",
        "饮品": "饮品/甜点",
        "奶茶": "饮品/甜点",
        "超市": "超市/零售",
        "便利": "超市/零售",
        "商店": "超市/零售",
        "书店": "文创/书店",
    }
    for keyword, category in mapping.items():
        if keyword in name:
            return category
    return "其他"


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """个人信息查看和编辑页面"""
    user = request.user
    form = ProfileForm(request.POST or None, request.FILES or None, instance=user)
    
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "个人信息已更新。")
        return redirect("MyApp:profile")
    
    # 统计用户相关数据
    user_consumption_count = ConsumptionRecord.objects.filter(user=user).count()
    user_total_amount = (
        ConsumptionRecord.objects.filter(user=user).aggregate(total=Sum("amount"))["total"]
        or Decimal("0")
    )
    
    return render(
        request,
        "profile.html",
        {
            "form": form,
            "user": user,
            "user_consumption_count": user_consumption_count,
            "user_total_amount": user_total_amount,
        },
    )


@login_required
def model_analysis_view(request: HttpRequest) -> HttpResponse:
    """模型分析页面：K-Means聚类和孤立森林异常检测"""
    try:
        # 检查是否有数据
        total_records = ConsumptionRecord.objects.count()
        if total_records == 0:
            messages.warning(request, "暂无消费数据，请先上传数据。")
            return render(request, "model_analysis.html", {"has_data": False})

        # 获取所有学生的消费数据并计算特征
        student_stats = (
            ConsumptionRecord.objects.values("student_no")
            .annotate(
                total_amount=Sum("amount"),
                avg_amount=Avg("amount"),
                max_amount=Max("amount"),
                min_amount=Min("amount"),
                transaction_count=Count("id"),
            )
            .order_by("student_no")
        )

        if not student_stats:
            messages.warning(request, "暂无学生消费数据。")
            return render(request, "model_analysis.html", {"has_data": False})

        # 准备特征数据
        student_features = []
        student_nos = []
        
        for stat in student_stats:
            student_no = stat["student_no"]
            student_nos.append(student_no)
            
            # 获取该学生的详细消费记录
            records = ConsumptionRecord.objects.filter(student_no=student_no)
            
            # 计算时间特征
            hours = [h for h in records.annotate(hour=ExtractHour("transaction_time")).values_list("hour", flat=True) if h is not None]
            weekdays = [wd for wd in records.annotate(weekday=ExtractWeekDay("transaction_time")).values_list("weekday", flat=True) if wd is not None]
            
            # 工作日消费比例（周一到周五，Django的weekday: 1=Monday, 7=Sunday）
            weekday_ratio = sum(1 for wd in weekdays if 1 <= wd <= 5) / len(weekdays) if weekdays else 0
            
            # 平均消费时间（小时）
            avg_hour = np.mean(hours) if hours else 12
            
            # 消费时间标准差（反映消费时间规律性）
            hour_std = np.std(hours) if hours else 0
            
            # 商户多样性（不同商户数量）
            merchant_diversity = records.values("merchant_name").distinct().count()
            
            # 构建特征向量（每个学生都会被构造成一个 8 维特征向量）
            features = [
                float(stat["total_amount"] or 0),  # 总消费金额
                float(stat["avg_amount"] or 0),    # 平均消费金额
                float(stat["max_amount"] or 0),    # 最大单笔消费
                stat["transaction_count"],          # 消费频次
                merchant_diversity,                 # 商户多样性
                weekday_ratio,                     # 工作日消费比例
                avg_hour,                          # 平均消费时间
                hour_std,                          # 消费时间标准差
            ]
            student_features.append(features)

        if len(student_features) < 2:
            messages.warning(request, "学生数量不足，无法进行聚类分析（至少需要2个学生）。")
            return render(request, "model_analysis.html", {"has_data": False})

        # 转换为numpy数组（所有学生的数据会变成一个矩阵）
        X = np.array(student_features)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means聚类
        n_clusters = min(5, len(student_features))  # 最多5个聚类
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        def mask_student_no(student_no):   # 学工号脱敏处理（中间部分用*代替）
            # 确保 student_no 长度足够，避免索引超出
            if len(student_no) > 8:
                return student_no[:4] + '*' * (len(student_no) - 8) + student_no[-4:]
            return student_no
        # 准备聚类结果
        cluster_results = []
        for i in range(n_clusters):
            cluster_students = [student_nos[j] for j in range(len(student_nos)) if cluster_labels[j] == i]
            cluster_data = [
                {
                    "student_no": student_nos[j]    ,
                    "total_amount": float(student_features[j][0]),
                    "avg_amount": float(student_features[j][1]),
                    "transaction_count": int(student_features[j][3]),
                }
                for j in range(len(student_nos))
                if cluster_labels[j] == i
            ]
            cluster_results.append({
                "cluster_id": i,
                "student_count": len(cluster_students),
                "students": cluster_students[:20],  # 最多显示20个
                "total_students": len(cluster_students),
                "cluster_data": cluster_data,
                "avg_total_amount": np.mean([f[0] for f, label in zip(student_features, cluster_labels) if label == i]) if any(cluster_labels == i) else 0,
                "avg_transaction_count": np.mean([f[3] for f, label in zip(student_features, cluster_labels) if label == i]) if any(cluster_labels == i) else 0,
            })

        # 孤立森林异常检测
        contamination = min(0.1, max(0.05, 5 / len(student_features)))  # 异常比例，最多10%
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = isolation_forest.fit_predict(X_scaled) #得到异常或正常的标签
        
        # 准备异常检测结果
        anomaly_results = []
        normal_results = []

        for i, (student_no, features, is_anomaly) in enumerate(zip(student_nos, student_features, anomaly_labels)):

            result_item = {
                "student_no": student_no,
                "total_amount": float(features[0]),
                "avg_amount": float(features[1]),
                "max_amount": float(features[2]),
                "transaction_count": int(features[3]),
                "merchant_diversity": int(features[4]),
                "weekday_ratio": float(features[5]),
                "anomaly_score": float(isolation_forest.score_samples(X_scaled[i:i+1])[0]),
                "cluster": int(cluster_labels[i]),
            }
            
            if is_anomaly == -1:  # 异常
                anomaly_results.append(result_item)
            else:  # 正常
                normal_results.append(result_item)
        
        # 按异常分数排序（分数越低越异常）
        anomaly_results.sort(key=lambda x: x["anomaly_score"])
        
        # 准备图表数据
        # 聚类分布图
        cluster_distribution = {
            "labels": [f"聚类 {i+1}" for i in range(n_clusters)],
            "counts": [cr["student_count"] for cr in cluster_results],
        }
        
        # 异常检测统计
        anomaly_stats = {
            "total_students": len(student_nos),
            "anomaly_count": len(anomaly_results),
            "normal_count": len(normal_results),
            "anomaly_ratio": len(anomaly_results) / len(student_nos) * 100 if student_nos else 0,
        }



        for record in anomaly_results:
            record["student_no"] = mask_student_no(record["student_no"])
        for record in normal_results:
            record["student_no"] = mask_student_no(record["student_no"])
        print(cluster_results)
        return render(
            request,
            "model_analysis.html",
            {
                "has_data": True,
                "cluster_results": cluster_results,
                "anomaly_results": anomaly_results[:50],  # 最多显示50个异常
                "normal_results": normal_results[:20],  # 显示部分正常样本
                "cluster_distribution": json.dumps(cluster_distribution, ensure_ascii=False),
                "anomaly_stats": anomaly_stats,
                "total_students": len(student_nos),
                "feature_names": [
                    "总消费金额",
                    "平均消费金额",
                    "最大单笔消费",
                    "消费频次",
                    "商户多样性",
                    "工作日消费比例",
                    "平均消费时间",
                    "消费时间标准差",
                ],
            },
        )
    
    except ImportError:
        messages.error(
            request,
            "缺少必要的Python库。请安装：pip install scikit-learn numpy"
        )
        return render(request, "model_analysis.html", {"has_data": False, "error": "missing_library"})
    except Exception as e:
        messages.error(request, f"模型分析出错：{str(e)}")
        return render(request, "model_analysis.html", {"has_data": False, "error": str(e)})

