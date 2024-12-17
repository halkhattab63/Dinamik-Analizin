import pstats

# تحديد الملف
file_path = "profile.stats"

# تحميل ملف الأداء
stats = pstats.Stats(file_path)

# فرز البيانات حسب الزمن المستغرق في الدوال
stats.sort_stats("cumulative")

# عرض أفضل 20 دالة
stats.print_stats(20)
