"""Command-line interface orchestration for the Formative demo."""
from __future__ import annotations

import argparse
from typing import Optional

from .face import load_face_model, predict_face
from .metadata import load_class_names
from .paths import MODEL_DIR
from .recommender import generate_recommendations, load_recommender_pipeline, recommend_stub
from .voice import VOICE_PROB_THRESHOLD, load_audio_features, load_voice_assets, predict_voice


def run_demo(
    face_path: Optional[str],
    voice_filename: Optional[str],
    *,
    recommender_customer_id: Optional[str] = None,
    merged_csv: Optional[str] = None,
    interactive: bool = False,
) -> None:
    if interactive:
        while not face_path:
            face_path = input("Enter face image path: ").strip() or None
    if not face_path:
        print("Face image path is required.")
        return

    print("Loading class names...")
    class_names = load_class_names()

    print("Loading face model...")
    face_model = load_face_model()

    print("Running face authentication...")
    try:
        face_ok, face_name, face_score = predict_face(face_model, face_path, class_names)
    except Exception as exc:
        print("Face authentication failed:", exc)
        return
    print(f"Face result: {face_name} ({face_score:.2f}%) -> {'PASS' if face_ok else 'DENY'}")
    face_name_str = str(face_name).strip().lower()
    if face_name_str == "unknown":
        print("Face matched an unknown identity.")
        face_ok = False
    if not face_ok:
        print("ACCESS DENIED (face)")
        return

    if interactive and recommender_customer_id is None:
        user_input = input("Enter customer ID for recommendation (leave blank to skip): ").strip()
        recommender_customer_id = user_input or None
    if interactive and recommender_customer_id and merged_csv is None:
        merged_input = input("Enter merged CSV path (leave blank to skip recommendations): ").strip()
        merged_csv = merged_input or None

    print("Running product recommendation model...")
    pipeline = load_recommender_pipeline()
    recommendations = recommend_stub()
    used_stub = True
    if pipeline and recommender_customer_id and merged_csv:
        try:
            recommendations = generate_recommendations(
                pipeline,
                merged_csv,
                recommender_customer_id,
            )
            used_stub = False
        except Exception as exc:
            print("Recommender pipeline error:", exc)
            recommendations = recommend_stub()
            used_stub = True
    else:
        if (MODEL_DIR / "recommender_pipeline.pkl").exists():
            print(
                "Recommender pipeline found. To get real recommendations pass customer ID and merged CSV."
            )
        else:
            print("No recommender pipeline found. Using demo recommendations.")

    if interactive and not voice_filename:
        voice_filename = input("Enter voice filename: ").strip() or None
    if not voice_filename:
        print("Voice filename is required.")
        return

    print("Running voice validation model...")
    print("Loading audio features...")
    audio_df = load_audio_features()
    print("Loading voice assets...")
    try:
        voice_assets = load_voice_assets()
    except Exception as exc:
        print("Voice asset load failed:", exc)
        return

    voice_model = voice_assets["model"]
    voice_encoder = voice_assets["encoder"]
    voice_mode = voice_assets["mode"]
    scaler = voice_assets.get("scaler")
    feature_columns = voice_assets.get("feature_columns")

    if voice_mode == "mlp" and scaler is None:
        print("Voice scaler missing for v2 model; cannot continue.")
        return

    try:
        vidx, vprob = predict_voice(
            voice_model,
            audio_df,
            voice_filename,
            scaler=scaler,
            mode=voice_mode,
            feature_columns=feature_columns,
        )
    except Exception as exc:
        print("Voice verification failed:", exc)
        return

    speaker_name = None
    if voice_encoder is not None:
        try:
            speaker_name = voice_encoder.inverse_transform([vidx])[0]
        except Exception:
            speaker_name = None

    speaker_label = speaker_name.strip().lower() if isinstance(speaker_name, str) else ""
    if speaker_label == "unknown":
        print("Voice matched an unknown speaker.")
        vprob = 0.0

    voice_display = speaker_name if speaker_name else f"class_index={vidx}"
    status = "PASS" if vprob >= VOICE_PROB_THRESHOLD else "DENY"
    print(f"Voice result[{voice_mode}]: {voice_display} -> {status}")
    if vprob < VOICE_PROB_THRESHOLD:
        print("ACCESS DENIED (voice)")
        return

    if class_names and speaker_name and speaker_name.strip().lower() != "unknown":
        face_identity = str(face_name).strip().lower()
        voice_identity = str(speaker_name).strip().lower()
        if face_identity != voice_identity:
            print(f"Multimodal mismatch: face={face_name} vs voice={speaker_name}.")
            print("ACCESS DENIED (identity mismatch)")
            return

    print("\nTRANSACTION RESULT: ACCESS GRANTED")
    if used_stub:
        print("Recommended products (demo mode):")
    else:
        print("Recommended products (personalized):")
    for rec in recommendations:
        print(f" - {rec['name']} (id={rec['product_id']}, score={rec['score']})")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo CLI: face -> voice -> recommender")
    parser.add_argument("--face", required=False, help="Path to face image (jpg/png)")
    parser.add_argument(
        "--voice",
        required=False,
        help="Audio filename",
    )
    parser.add_argument(
        "--recommender-customer-id",
        required=False,
        help="Customer id to use for recommender",
    )
    parser.add_argument(
        "--merged-csv",
        required=False,
        help="Path or URL to merged CSV with customer features for recommender",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for inputs in sequential order",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = parse_args()
    args = parser.parse_args(argv)
    interactive = args.interactive or not (args.face and args.voice)
    if not interactive and (not args.face or not args.voice):
        print("Face image and voice filename are required unless --interactive is used.")
        return

    run_demo(
        args.face,
        args.voice,
        recommender_customer_id=args.recommender_customer_id,
        merged_csv=args.merged_csv,
        interactive=interactive,
    )
