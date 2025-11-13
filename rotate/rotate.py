#!/usr/bin/env python3
"""
간단한 명령줄 영상 회전 도구
"""
import subprocess
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("=" * 60)
        print("           FFmpeg 영상 회전 도구")
        print("=" * 60)
        print()
        print("사용법:")
        print(f"  python {sys.argv[0]} <입력파일> <출력파일> [회전방향]")
        print()
        print("예시:")
        print(f"  python {sys.argv[0]} input.mp4 output.mp4 ccw_90")
        print(f"  python {sys.argv[0]} input.mp4 output.mp4 cw_90")
        print(f"  python {sys.argv[0]} input.mp4 output.mp4 180")
        print()
        print("회전 방향 (기본값: ccw_90):")
        print("  ccw_90   : 반시계방향 90도 ↺ (세로 → 가로)")
        print("  cw_90    : 시계방향 90도 ↻ (가로 → 세로)")
        print("  180      : 180도 회전 (상하좌우 반전)")
        print("  ccw_270  : 반시계방향 270도")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    rotation = sys.argv[3] if len(sys.argv) > 3 else "ccw_90"
    
    # 회전 필터 매핑
    rotation_filters = {
        "ccw_90": "transpose=2",
        "cw_90": "transpose=1",
        "180": "transpose=2,transpose=2",
        "ccw_270": "transpose=1",
    }
    
    rotation_names = {
        "ccw_90": "반시계방향 90도 ↺",
        "cw_90": "시계방향 90도 ↻",
        "180": "180도 회전",
        "ccw_270": "반시계방향 270도"
    }
    
    if rotation not in rotation_filters:
        print(f"❌ 오류: 지원하지 않는 회전 방향입니다: {rotation}")
        print(f"지원하는 옵션: {', '.join(rotation_filters.keys())}")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)
    
    print(f"📹 입력: {input_file}")
    print(f"💾 출력: {output_file}")
    print(f"🔄 회전: {rotation_names[rotation]}")
    print()
    print("⚙️  처리 중... (회전은 재인코딩이 필요하므로 시간이 걸립니다)")
    
    # ffmpeg 명령 실행
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vf", rotation_filters[rotation],
        "-c:a", "copy",
        "-y",
        output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print()
            print("✅ 완료!")
            print(f"📁 {os.path.abspath(output_file)}")
        else:
            print()
            print("❌ 오류 발생:")
            print(result.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print("❌ ffmpeg가 설치되어 있지 않습니다!")
        print()
        print("설치 방법:")
        print("  Windows: https://ffmpeg.org/download.html")
        print("  Mac: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        sys.exit(1)

if __name__ == "__main__":
    main()