#!/usr/bin/env python3
"""
Finds the sync offset between two full audio files, then uses a known video
start time to calculate the precise start point for a new, synchronized WAV file,
and automatically generates the output file and a correlation graph.

Usage:
    python real.py /path/to/video.mp4 /path/to/separate.wav
"""
import sys
import os
import subprocess
import tempfile
import pickle
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# --- 設定項目 ---
CONFIG = {
    # ★★★★★【要設定】わかっているビデオの開始位置を「時:分:秒.ミリ秒」で指定 ★★★★★
    'KNOWN_VIDEO_START_TIME': '00:00:00.000',  # 例: '00:05:00.500' (5分0秒500ミリ秒)

    'sample_rate': 16000,
    'lowpass_freq': 4000,
}
# --- 設定項目ここまで ---


def parse_timecode_to_seconds(timecode: str) -> float:
    """
    HH:MM:SS.ms形式のタイムコード文字列を秒数（float）に変換します。
    """
    try:
        parts = timecode.split(':')
        if len(parts) != 3:
            raise ValueError("Timecode must be in HH:MM:SS.ms format")
            
        h = int(parts[0])
        m = int(parts[1])
        
        sec_ms_parts = parts[2].split('.')
        s = int(sec_ms_parts[0])
        ms = int(sec_ms_parts[1].ljust(3, '0')[:3]) if len(sec_ms_parts) > 1 else 0
        
        return h * 3600 + m * 60 + s + ms / 1000.0
    except (ValueError, IndexError):
        print(f"エラー: 無効なタイムコード形式です: '{timecode}'。'HH:MM:SS.ms' 形式で指定してください。")
        sys.exit(1)

def format_seconds_to_timecode(total_seconds: float) -> str:
    """
    秒数（float）をHH:MM:SS.ms形式のタイムコード文字列に変換します。
    """
    sign = ""
    if total_seconds < 0:
        sign = "-"
        total_seconds = abs(total_seconds)
        
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    ms = int(round((total_seconds - int(total_seconds)) * 1000))
    
    return f"{sign}{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class AudioSyncWavOutput:
    def __init__(self, video_path, audio_path):
        self.video_path = video_path
        self.audio_path = audio_path
        self.sr = CONFIG['sample_rate']
        self.cache_dir = Path(tempfile.gettempdir()) / "audio_sync_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def extract_audio_cached(self, input_path, tag=""):
        """キャッシュを利用して音声データを抽出する"""
        cache_key = f"{Path(input_path).stem}_{tag}_{self.sr}.pkl"
        cache_file = self.cache_dir / cache_key
        
        if cache_file.exists():
            print(f"キャッシュから音声を読み込み中: {cache_key}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        temp_wav = self.cache_dir / f"temp_{tag}_{hash(input_path)}.wav"
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-ac', '1', '-ar', str(self.sr), '-vn', str(temp_wav)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print("エラー: 音声抽出中にffmpegでエラーが発生しました。")
            print("--- ffmpeg error output ---\n{e.stderr}\n--------------------------")
            sys.exit(1)
        except FileNotFoundError:
            print("エラー: 'ffmpeg' コマンドが見つかりません。")
            sys.exit(1)

        rate, data = wavfile.read(temp_wav)
        
        if data.ndim > 1: data = data.mean(axis=1)
        data = data.astype(np.float32)
        
        with open(cache_file, 'wb') as f: pickle.dump((rate, data), f)
            
        return rate, data

    def preprocess_audio(self, audio_data):
        """音声データの前処理"""
        audio_data = audio_data - np.mean(audio_data)
        sos = butter(6, CONFIG['lowpass_freq'], 'low', fs=self.sr, output='sos')
        audio_data = sosfiltfilt(sos, audio_data)
        audio_data = audio_data / (np.percentile(np.abs(audio_data), 95) + 1e-8)
        audio_data = gaussian_filter1d(audio_data, sigma=1.0)
        return audio_data

    def find_offset_fast_correlation(self, ref_audio, search_audio):
        """FFTベースの相互相関を用いてオフセットを検出し、描画用のデータも返す"""
        print("音声全体の前処理を実行中...")
        ref_processed = self.preprocess_audio(ref_audio)
        search_processed = self.preprocess_audio(search_audio)
        
        print("FFT相互相関を計算中… (mode='full')")
        # full 相互相関を取ることで、ref が先行(負の lag)／後追い(正の lag)の両方を拾う
        correlation = correlate(search_processed, ref_processed, mode='full', method='fft')
        # lag の配列を作る（サンプル単位）
        lags = np.arange(-len(ref_processed) + 1, len(search_processed))
        # ピークを探してサンプルオフセットに変換
        best_idx       = np.argmax(correlation)
        best_lag       = lags[best_idx]
        offset_seconds = best_lag / self.sr
        # 秒単位のオフセット配列
        offsets        = lags / self.sr
        
        return offset_seconds, offsets, correlation

    def visualize_correlation(self, offsets, corrs):
        """相関結果を可視化する（ハイライトなし）"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.plot(offsets, corrs, color='blue', label='FFT Correlation', alpha=0.8)
        
        ax.set_title('Full Audio Correlation Analysis')
        ax.set_xlabel('Base Offset in Longer Audio (seconds)')
        ax.set_ylabel('Correlation')
        ax.legend()
        ax.grid(True, alpha=0.4)
        
        plt.tight_layout()
        return fig

def main():
    # コンソールから動画と音声のパスを入力
    video_input = input("動画ファイルのパスを入力してください: ")
    audio_input = input("音声ファイルのパスを入力してください: ")
    video_path = Path(video_input.strip())
    audio_path = Path(audio_input.strip())
    
    if not video_path.exists() or not audio_path.exists():
        print(f"エラー: 指定されたファイルが見つかりません。")
        if not video_path.exists(): print(f" - {video_path}")
        if not audio_path.exists(): print(f" - {audio_path}")
        sys.exit(1)

    sync = AudioSyncWavOutput(video_path, audio_path)
    
    video_start_time_str = CONFIG['KNOWN_VIDEO_START_TIME']
    video_start_time_sec = parse_timecode_to_seconds(video_start_time_str)

    print("--- 同期プロセス開始 ---")
    print(f"ビデオの既知の開始位置: {video_start_time_str} ({video_start_time_sec:.3f} 秒)")
    print("-" * 25)

    print("ビデオ音声全体を抽出中...")
    _, video_audio = sync.extract_audio_cached(video_path, "full_video")
    
    print("外部音声全体を抽出中...")
    _, external_audio = sync.extract_audio_cached(audio_path, "full_external")
    
    if len(video_audio) >= len(external_audio):
        search_audio, ref_audio = video_audio, external_audio
        longer_source_is_video = True
        print("情報: ビデオ音声が長いため、その中から外部音声を検索します。")
    else:
        search_audio, ref_audio = external_audio, video_audio
        longer_source_is_video = False
        print("情報: 外部音声が長いため、その中からビデオ音声を検索します。")
        
    base_offset, offsets, corrs = sync.find_offset_fast_correlation(
        ref_audio, search_audio
    )
    
    if longer_source_is_video:
        final_audio_start_time = video_start_time_sec - base_offset
        print(f"\n基本オフセット（ビデオ内での外部音声の開始位置）: {base_offset:.3f}秒")
    else:
        final_audio_start_time = base_offset + video_start_time_sec
        print(f"\n基本オフセット（外部音声内でのビデオ音声の開始位置）: {base_offset:.3f}秒")

    # 4. 可視化グラフを生成
    print("相関グラフを生成中...")
    fig_corr = sync.visualize_correlation(offsets, corrs)
    corr_plot_path = "correlation_analysis_full.png"
    fig_corr.savefig(corr_plot_path, dpi=150, bbox_inches='tight')
    print(f"相関分析グラフを保存しました: {corr_plot_path}")

    # 5. 結果と、同期済みWAVを生成するFFmpegコマンドを出力
    print("\n" + "="*60)
    print("同期処理完了")
    print("="*60)
    
    # オーディオ開始位置が負の場合、音声は0秒から開始し、動画をオーディオ開始位置に合わせて開始
    if final_audio_start_time < 0:
        print("情報: 計算された音声開始位置が負のため、音声は0秒から開始し、動画を音声開始位置に合わせて開始します。")
        audio_ss = 0.0
        shifted_video_start_sec = video_start_time_sec - final_audio_start_time
        video_ss_str = format_seconds_to_timecode(shifted_video_start_sec)
    else:
        audio_ss = final_audio_start_time
        video_ss_str = video_start_time_str

    final_audio_start_timecode = format_seconds_to_timecode(audio_ss)
    print(f"計算の結果、外部音声の開始位置: {final_audio_start_timecode}、動画の開始位置: {video_ss_str}")

    output_wav_filename = f"{audio_path.stem}_synchronized.wav"
    
    # --- WAVファイル生成コマンドの定義と実行 ---
    print("\n--- 同期済みWAVファイルをトリムして生成します ---")
    
    # 動画全体を処理（duration指定なし）
    command_list = [
        "ffmpeg", "-y",
        "-ss", f"{audio_ss:.3f}",
        "-i", str(audio_path.resolve()),
        "-c:a", "pcm_s16le",
        output_wav_filename
    ]
    print("実行コマンド:", " ".join(f'"{arg}"' if " " in arg else arg for arg in command_list))

    try:
        subprocess.run(command_list, check=True)
        print(f"\n成功: '{output_wav_filename}' が正常に作成されました。")
    except FileNotFoundError:
        print("\nエラー: 'ffmpeg' コマンドが見つかりません。")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nエラー: FFmpegの実行中にエラーが発生しました (終了コード: {e.returncode})。")
        sys.exit(1)

    # --- 動画を同じ開始位置から90分間トリムして出力 ---
    video_output_filename = f"{video_path.stem}_trimmed.mp4"
    print("\n--- 動画全体を出力します ---")
    video_command = [
        "ffmpeg", "-y",
        "-ss", video_ss_str,
        "-i", str(video_path.resolve()),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        video_output_filename
    ]
    print("実行コマンド:", " ".join(f'"{arg}"' if " " in arg else arg for arg in video_command))
    try:
        subprocess.run(video_command, check=True)
        print(f"\n成功: '{video_output_filename}' が正常に作成されました。")
    except FileNotFoundError:
        print("\nエラー: 'ffmpeg' コマンドが見つかりません。")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nエラー: FFmpegの実行中にエラーが発生しました (終了コード: {e.returncode})。")
        sys.exit(1)


if __name__ == '__main__':
    main()
