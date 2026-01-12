
from PIL import Image, ImageDraw, ImageFont
import io
import aiohttp
import os

class MemeService:
    def __init__(self):
        pass

    async def get_avatar(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(str(url)) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    return Image.open(io.BytesIO(data))
        return None

    async def create_clown_license(self, user_name, avatar_url, issuer="Savage AI"):
        # Create base card (Landscape: 600x400)
        width, height = 600, 400
        card = Image.new('RGB', (width, height), color='#fdfbf7') # Off-white paper
        draw = ImageDraw.Draw(card)

        # Border
        draw.rectangle([10, 10, width-10, height-10], outline='#ef4444', width=5)
        draw.rectangle([15, 15, width-15, height-15], outline='#000000', width=2)
        
        # Header
        try:
            # Try to use a decent font if available, else default
            # Windows usually has arial.ttf
            header_font = ImageFont.truetype("arial.ttf", 40)
            text_font = ImageFont.truetype("arial.ttf", 20)
            big_font = ImageFont.truetype("arial.ttf", 30)
        except:
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            big_font = ImageFont.load_default()

        draw.text((300, 50), "OFFICIAL CLOWN LICENSE", fill='#ef4444', font=header_font, anchor="mm")
        
        # Avatar
        avatar = await self.get_avatar(avatar_url)
        if avatar:
            avatar = avatar.resize((150, 150))
            card.paste(avatar, (50, 100))
            # Draw border around avatar
            draw.rectangle([50, 100, 200, 250], outline='black', width=3)
        
        # Details
        draw.text((230, 120), f"Name: {user_name}", fill='black', font=big_font)
        draw.text((230, 170), f"ID: 69-420-CLOWN", fill='black', font=text_font)
        draw.text((230, 200), f"Issued By: {issuer}", fill='black', font=text_font)
        draw.text((230, 230), f"Expires: NEVER", fill='#ef4444', font=text_font)
        
        # Stamp
        draw.text((450, 300), "VERIFIED\nCLOWN", fill='#ef4444', font=header_font, anchor="mm", align='center')
        draw.rectangle([350, 250, 550, 350], outline='#ef4444', width=5)

        # Output
        buffer = io.BytesIO()
        card.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer

meme_service = MemeService()
