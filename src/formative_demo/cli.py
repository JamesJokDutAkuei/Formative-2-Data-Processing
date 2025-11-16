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
    face_path: str,
    voice_filename: str,
    *,
    recommender_customer_id: Optional[str] = None,
    merged_csv: Optional[str] = None,
) -> None:
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
    if not face_ok:
        print("ACCESS DENIED (face)")
        return

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

    voice_display = speaker_name if speaker_name else f"class_index={vidx}"
    status = "PASS" if vprob >= VOICE_PROB_THRESHOLD else "DENY"
    print(f"Voice result[{voice_mode}]: {voice_display}, prob={vprob:.3f} -> {status}")
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

    pipeline = load_recommender_pipeline()
    if pipeline and recommender_customer_id and merged_csv:
        print("Loading recommender pipeline...")
        try:
            recommendations = generate_recommendations(
                pipeline,
                merged_csv,
                recommender_customer_id,
            )
        except Exception as exc:
            print("Recommender pipeline error:", exc)
            recommendations = recommend_stub()
    else:
        if (MODEL_DIR / "recommender_pipeline.pkl").exists():
            print(
                "Recommender pipeline found. To get real recommendations pass `--recommender-customer-id`"
                " and `--merged-csv <path_or_url>` to the CLI."
            )
        else:
            print("No recommender pipeline found. Returning demo recommendations.")
        recommendations = recommend_stub()

    print("\nTRANSACTION RESULT: ACCESS GRANTED")
    print("Recommended products:")
    for rec in recommendations:
        print(f" - {rec['name']} (id={rec['product_id']}, score={rec['score']})")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo CLI: face -> voice -> recommender")
    parser.add_argument("--face", required=True, help="Path to face image (jpg/png)")
    parser.add_argument(
        "--voice",
        required=True,
        help="Audio filename (must match `filename` in audio_features.csv`)",
    )
    parser.add_argument(
        "--recommender-customer-id",
        required=False,
        help="Customer id to use for recommender (must exist in merged CSV)",
    )
    parser.add_argument(
        "--merged-csv",
        required=False,
        help="Path or URL to merged CSV with customer features for recommender",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = parse_args()
    args = parser.parse_args(argv)
    run_demo(
        args.face,
        args.voice,
        recommender_customer_id=args.recommender_customer_id,
        merged_csv=args.merged_csv,
    )
