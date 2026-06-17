from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

def fetch_video_info_with_comments(url, max_comments=5):
    """
    Extracts video description + top N comments using yt-dlp.

    Parameters:
        url (str): YouTube video URL
        max_comments (int): Number of comments to return

    Returns:
        dict: {
            "description": str,
            "comments": list[dict]
        }
    """

    ydl_opts = {
        "skip_download": True,
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                "max_comments": [str(max_comments)],
                "comment_sort": ["top"],
            }
        }
    }

    ##skip authentication for now
    ##implement your own secure connection
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except DownloadError as e:
            if "Sign in to confirm your age" in str(e):
                return {
                    "description": "",
                    "comments": []
                }
            raise

    description = info.get("description", "")
    comments = (info.get("comments") or [])[:max_comments]

    return {
        "description": description,
        "comments": comments
    }
