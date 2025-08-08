#!/usr/bin/env python3
"""
Telegram Bot Setup Helper

This script helps you:
1. Test your bot token
2. Get your chat ID
3. Test bot functionality
"""

import sys
import os
import asyncio
from telegram import Bot
from telegram.error import TelegramError

# Add project root to path
sys.path.insert(0, '.')

async def test_bot_token(token: str):
    """Test if the bot token is valid."""
    try:
        bot = Bot(token=token)
        bot_info = await bot.get_me()
        print(f"✅ Bot token is valid!")
        print(f"   Bot name: {bot_info.first_name}")
        print(f"   Bot username: @{bot_info.username}")
        return bot
    except TelegramError as e:
        print(f"❌ Bot token error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def get_chat_id_instructions(bot: Bot):
    """Provide instructions to get chat ID."""
    print("\n📱 TO GET YOUR CHAT ID:")
    print("=" * 40)
    print("1. Open Telegram on your phone/computer")
    print(f"2. Search for your bot: @{(await bot.get_me()).username}")
    print("3. Start a conversation with your bot by sending: /start")
    print("4. Send any message to your bot (like 'hello')")
    print("5. Then run this command to get your chat ID:")
    print(f"   curl https://api.telegram.org/bot{bot.token}/getUpdates")
    print("\n6. Look for 'chat':{'id': YOUR_CHAT_ID} in the response")
    print("7. Update your .env file with: TELEGRAM_CHAT_ID=your_chat_id")

async def test_with_chat_id(bot: Bot, chat_id: str):
    """Test sending a message with chat ID."""
    try:
        message = """
🧪 TELEGRAM BOT TEST

✅ Your algo trading bot is working!

This confirms:
• Bot token is valid
• Chat ID is correct
• Message delivery works

Your trading alerts are ready! 🚀
        """
        
        await bot.send_message(
            chat_id=chat_id,
            text=message.strip(),
            parse_mode='HTML'
        )
        print(f"✅ Test message sent successfully to chat ID: {chat_id}")
        return True
        
    except TelegramError as e:
        print(f"❌ Error sending message: {e}")
        return False

async def main():
    """Main function."""
    print("🤖 TELEGRAM BOT SETUP HELPER")
    print("=" * 40)
    
    # Load bot token from .env
    token = "8020776356:AAF1EwkNW1RN3y1XVmP675xH8dKWInPwrrE"
    
    print(f"🔑 Testing bot token...")
    bot = await test_bot_token(token)
    
    if not bot:
        print("\n❌ Bot token is invalid. Please check your token.")
        return 1
    
    # Check if chat ID is provided
    chat_id = input("\n💬 Enter your chat ID (press Enter if you don't have it yet): ").strip()
    
    if chat_id:
        print(f"\n📤 Testing message delivery to chat ID: {chat_id}")
        success = await test_with_chat_id(bot, chat_id)
        
        if success:
            print("\n🎉 SETUP COMPLETE!")
            print("Your Telegram bot is ready for trading alerts!")
            
            # Update .env file
            with open('.env', 'r') as f:
                content = f.read()
            
            if 'TELEGRAM_CHAT_ID=' in content:
                # Update existing line
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('TELEGRAM_CHAT_ID=') or line.startswith('# TELEGRAM_CHAT_ID='):
                        lines[i] = f'TELEGRAM_CHAT_ID={chat_id}'
                        break
                content = '\n'.join(lines)
            else:
                # Add new line
                content += f'\nTELEGRAM_CHAT_ID={chat_id}\n'
            
            with open('.env', 'w') as f:
                f.write(content)
            
            print(f"✅ Chat ID saved to .env file")
        else:
            print("\n❌ Message delivery failed. Please check your chat ID.")
            await get_chat_id_instructions(bot)
    else:
        await get_chat_id_instructions(bot)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled.")
        sys.exit(0)
