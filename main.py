ai-painting
main.py
import base64
import io
import json
import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, ImageEnhance

# --- AI CONFIGURATION ---
USE_REAL_AI = True 
pipe = None

async def load_models():
    """Loads the AI models asynchronously on startup."""
    global pipe, USE_REAL_AI
    
    if not USE_REAL_AI:
        print("ℹ️  AI Disabled: Running in Simulation Mode.")
        return

    try:
        # --- DEPENDENCY CHECK ---
        try:
            import peft
        except ImportError:
            print("\n" + "="*50)
            print("❌ ERROR: The 'peft' library is missing!")
            print("   This is required for the Turbo/LoRA features.")
            print("   👉 TO FIX: Run this command in your terminal:")
            print("   pip install peft")
            print("="*50 + "\n")
            USE_REAL_AI = False
            return

        import torch
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
        
        # Check for CUDA availability
        if not torch.cuda.is_available():
            print("\n" + "="*50)
            print("⚠️  WARNING: GPU NOT DETECTED! Running on CPU.")
            print("   The generation will be slow (5-10 seconds).")
            print("   👉 TO FIX: Run this command in your terminal:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("="*50 + "\n")
        else:
            print(f"\n✅  GPU DETECTED: {torch.cuda.get_device_name(0)}")

        print("⏳ Loading ControlNet (Smarter Scribble v1.1)...")
        # Try loading online, fallback to offline if DNS/Net fails
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_scribble",
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        except Exception as e:
            print(f"   ⚠️ ControlNet Connection error: {e}")
            print("   🔄 Switching to OFFLINE mode for ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_scribble",
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True
            )

        print("⏳ Loading DreamShaper 8 (High Quality Engine)...")
        try:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "Lykon/dreamshaper-8",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                use_safetensors=True
            )
        except Exception as e:
            print(f"   ⚠️ DreamShaper Connection error: {e}")
            print("   🔄 Switching to OFFLINE mode for DreamShaper...")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "Lykon/dreamshaper-8",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                use_safetensors=True,
                local_files_only=True
            )

        print("⏳ Loading LCM-LoRA (Turbo Engine)...")
        try:
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        except Exception as e:
            print(f"   ⚠️ LoRA Connection/Load error: {e}")
            print("   🔄 Switching to OFFLINE mode for LoRA...")
            # Fallback to local cache if internet is down
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", local_files_only=True)

        pipe.fuse_lora() # CRITICAL: Merge weights for faster inference
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        if torch.cuda.is_available():
            # Load directly to GPU for instant generation
            pipe.to("cuda")
            print("✅ Models loaded: DreamShaper Ultra-Turbo (2-Step Mode)")
        else:
            pipe.to("cpu")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("   If this is a 'local_files_only' error, it means you haven't downloaded the models yet.")
        print("   Please check your internet connection and try again once to download them.")
        USE_REAL_AI = False

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await load_models()

def process_image(image_data: str, prompt: str):
    """Decodes base64 sketch -> Runs AI -> Returns base64 image."""
    try:
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        # 1. Decode
        input_image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        input_image = input_image.resize((512, 512))
        
        # 2. PREPROCESSING FOR "VARIOUS PENS"
        # Convert to Grayscale (L) first. 
        # This treats Red/Blue/Green pens as standard structure lines.
        input_image = input_image.convert("L")
        
        # 3. ENHANCE LINE VISIBILITY
        # Boost contrast carefully (2.0) so lines are clear but not too jagged/distorted
        enhancer = ImageEnhance.Contrast(input_image)
        input_image = enhancer.enhance(2.0) 
        
        # 4. Invert colors (Black on White -> White on Black)
        # ControlNet expects white lines on black background
        input_image = ImageOps.invert(input_image)
        
        # Convert back to RGB for the pipeline
        input_image = input_image.convert("RGB")
        
        # 5. Generate
        if USE_REAL_AI and pipe:
            user_prompt = prompt.strip()
            if not user_prompt:
                # Better default than "natural subject" which triggers faces
                user_prompt = "Scenery"
            
            # FORCE REALISM STYLE - BRIGHTER & MORE NATURAL
            style_prompt = (
                "Scenery, highly detailed, photorealistic, 8k, ultra realistic, "
                "soft natural daylight, bright, clear details, photography, "
                "vibrant colors, HDR"
            )
            final_prompt = f"{user_prompt}, {style_prompt}"
            
            # SMART NEGATIVE PROMPT
            # Standard cleanup + dark/moody removal
            negative_prompt = "cartoon, sketch, pencil drawing, grayscale, dull, blur, text, watermark, low quality, ugly, deformed, dark, moody, night, monochrome"
            
            # Detect if user wants a person. If NOT, strictly ban humans.
            # This fixes the "one line becomes a face" issue.
            user_prompt_lower = user_prompt.lower()
            human_keywords = ["man", "woman", "person", "human", "face", "boy", "girl", "portrait", "people", "child", "eye", "mouth"]
            
            if not any(k in user_prompt_lower for k in human_keywords):
                # Apply strong negative weighting against humans
                negative_prompt += ", (human:2.0), (face:2.0), (person:2.0), (people:2.0), (skin:1.5)"

            # Fixed seed prevents flickering
            import torch
            generator = torch.manual_seed(42) 
            
            output = pipe(
                final_prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                num_inference_steps=4,         # Quality: 4 Steps (Better texture than 2)
                guidance_scale=1.0,            # Adherence: 1.5 forces photorealism
                controlnet_conditioning_scale=1.0, # Flexibility: 1.0 allows for better interpretation
                cross_attention_kwargs={"scale": 1.0},
                generator=generator
            ).images[0]
            
        else:
            # Simulation Mode
            output = ImageOps.invert(input_image)

        # 6. Encode
        buffered = io.BytesIO()
        output.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.get("/")
async def get():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: index.html not found</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            # Updated logic: Check for 'image' field directly to handle any image payload
            # regardless of event type name (stroke vs image vs drawing)
            if payload.get("image"):
                prompt = payload.get("prompt", "")
                image_base64 = payload.get("image")

                loop = asyncio.get_event_loop()
                result_image = await loop.run_in_executor(None, process_image, image_base64, prompt)

                if result_image:
                    await websocket.send_json({
                        "type": "result", 
                        "image": result_image
                    })

    except WebSocketDisconnect:
        print("Client disconnected")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
