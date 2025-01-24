from datetime import datetime, timedelta, timezone


def gps_to_utc(gps_week, gps_seconds):
    # GPS周起始时间
    gps_epoch = datetime(1980, 1, 6)
    # 计算GPS周对应的总天数
    total_days = gps_week * 7
    # 计算GPS秒对应的时间差, GPS时间与UTC时间在秒上相差18秒
    time_difference = timedelta(days=total_days, seconds=gps_seconds-18)
    # 计算UTC时间
    utc_time = gps_epoch + time_difference
    return utc_time


def utc_to_gps(utc_time: datetime | None = None):
    utc_time = utc_time if utc_time is not None else datetime.now(timezone.utc)
    utc_time = utc_time.replace(tzinfo=None)
    # GPS周起始时间
    gps_epoch = datetime(1980, 1, 6)
    # 计算时间差
    time_difference = utc_time - gps_epoch
    # 计算总秒数
    total_seconds = time_difference.total_seconds()
    # print(total_seconds)
    # 计算GPS周数和周内秒
    gps_week = int(total_seconds // (7 * 24 * 3600))
    gps_seconds = int(total_seconds % (7 * 24 * 3600))
    gps_milliseconds = int((total_seconds % 1)*1e3)
    return gps_week, gps_seconds, gps_milliseconds
