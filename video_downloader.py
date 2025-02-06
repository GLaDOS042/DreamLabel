import os
from tqdm import tqdm
import yt_dlp

def download_video(url: str | list, output_path: str) -> None:

    if url is None or url == "" or len(url) == 0:
        return

    if os.path.exists(output_path):
        print(f"Video already exists at {output_path}")
        return
    
    with tqdm(total=0, unit='B', unit_scale=True, desc="Downloading video") as pbar:
        def progress_hook(d):
            if d['status'] != 'downloading':
                if d['status'] == 'finished':
                    pbar.close()
                    print("Download complete")
                return

            if d.get('total_bytes'):
                pbar.total = d['total_bytes']
            downloaded = d.get('downloaded_bytes', 0)
            pbar.update(downloaded - pbar.n)

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(os.path.dirname(output_path), '%(id)s.%(ext)s'),
            'progress_hooks': [progress_hook],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if isinstance(url, str):
                url = [url]
            for video_url in url:
                print(f"Downloading: {video_url}")
                info_dict = ydl.extract_info(video_url, download=True)