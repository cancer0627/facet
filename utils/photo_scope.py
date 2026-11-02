"""升级任务的照片范围过滤，使用单个参数传递整批路径。"""
import json


def photo_scope_sql(photo_path, column="path"):
    """返回 SQL（结构化查询语言）片段与参数，兼容单张和整批范围。"""
    if photo_path is None:
        return "", ()
    if isinstance(photo_path, str):
        return f" AND {column} = ?", (photo_path,)
    # json_each 避免大图库触发数据库绑定参数数量上限；空列表不匹配任何照片。
    return f" AND {column} IN (SELECT value FROM json_each(?))", (json.dumps(photo_path),)
