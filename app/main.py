import io
from typing import List
import make87
from PIL import Image
from make87_messages.core.header_pb2 import Header
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from make87_messages.text.text_plain_pb2 import PlainText
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
import torch
from google.protobuf.timestamp_pb2 import Timestamp


def main():
    make87.initialize()
    image_stream = make87.get_subscriber(name="IMAGE_STREAM", message_type=ImageJPEG)
    description_stream = make87.get_publisher(name="IMAGE_DESCRIPTION", message_type=PlainText)

    image_buffer = []  # Buffer to store frames
    ts_buffer = []  # Buffer to store timestamps
    window_size = make87.get_config_value("FRAME_WINDOW_SIZE", default=10, decode=int)

    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2"
    ).to("cuda")

    def process_frames(frames: List[Image], timestamps: List[int]):
        nonlocal processor, model
        # Placeholder for processing logic
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in frames],
                    {"type": "text", "text": "Describe what is happening in these frames."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        print(generated_texts[0])

        average_ts = sum(timestamps) / len(timestamps)

        header = Header(timestamp=Timestamp())
        header.timestamp.FromMicroseconds(int(average_ts * 1e6))  # Convert to microseconds
        description_stream.publish(PlainText(header=header, body=generated_texts[0]))

    def callback(message: ImageJPEG):
        nonlocal image_buffer, ts_buffer, window_size
        image = Image.open(io.BytesIO(message.data)).convert("RGB")
        image_buffer.append(image)  # Add the new frame to the buffer
        ts_buffer.append(message.header.timestamp.ToDatetime().timestamp())

        # If the buffer has at least `window_size` frames, process the sliding window
        if len(image_buffer) >= window_size:
            process_frames(
                image_buffer[:window_size], ts_buffer[:window_size]
            )  # Process the first `window_size` frames
            image_buffer = image_buffer[1:]  # Remove the oldest frame to maintain the sliding window
            ts_buffer = ts_buffer[1:]  # Remove the oldest timestamp

    image_stream.subscribe(callback)
    make87.loop()


if __name__ == "__main__":
    main()
