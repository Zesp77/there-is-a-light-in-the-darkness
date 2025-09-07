"""
Telegram notification system for scalp trading signals

Sends formatted trade signals, updates, and alerts via Telegram bot
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional
import aiohttp
import json

from utils.logger import setup_logger

logger = setup_logger(__name__)

class TelegramNotifier:
    """
    Handles Telegram notifications for trading signals and updates
    """
    
    def __init__(self):
        """Initialize Telegram notifier"""
        self.bot_token = None
        self.chat_id = None
        self.enabled = False
        self.session = None
        
        # Message templates
        self.templates = {
            'startup': "🚀 **Lighter Scalp Bot Started**\n\nBot is now scanning for scalp opportunities...",
            
            'signal': """
🎯 **{quality} SIGNAL DETECTED**

📊 **Pair:** {pair}
📈 **Direction:** {side_emoji} {side_text}
💰 **Entry Price:** ${entry_price:.6f}

⚡ **Trade Parameters:**
• 🎯 TP1: ${tp1:.6f} ({tp1_pct:+.2f}%)
• 🎯 TP2: ${tp2:.6f} ({tp2_pct:+.2f}%)
• 🎯 TP3: ${tp3:.6f} ({tp3_pct:+.2f}%)
• 🛡️ Stop Loss: ${sl:.6f} ({sl_pct:.2f}%)
• ⚖️ Leverage: {leverage}x
• ⏰ Time Limit: {time_limit} min

📊 **Signal Metrics:**
• 🔥 Quality Score: {score:.1f}/100
• 📊 Volume Ratio: {volume:.1f}x
• 🌊 Momentum: {momentum:.0%}
• 📈 Volatility: {volatility:.2%}

💡 **Adjustments after 10 min if TP1 not hit:**
• Move SL to breakeven
• Consider partial exit at 50% of TP1

⏰ **Signal Time:** {timestamp}
            """,
            
            'update': """
🔄 **TRADE UPDATE**

📊 **Pair:** {pair}
💰 **Current Price:** ${current_price:.6f}
📈 **P&L:** {pnl:+.2f}% ({pnl_usd:+.2f} USD)

{update_message}

⏰ **Update Time:** {timestamp}
            """,
            
            'expired': """
⏰ **SIGNAL EXPIRED**

📊 **Pair:** {pair}
📈 **Direction:** {side_text}
💰 **Entry Price:** ${entry_price:.6f}

Signal time limit reached without execution.

⏰ **Expired at:** {timestamp}
            """
        }
        
        logger.info("📱 Telegram Notifier initialized")
    
    async def initialize(self, telegram_config: dict):
        """
        Initialize Telegram connection
        
        Args:
            telegram_config: Telegram configuration dict
        """
        try:
            self.bot_token = telegram_config.get('bot_token')
            self.chat_id = telegram_config.get('chat_id')
            self.enabled = telegram_config.get('enabled', False)
            
            if not self.enabled:
                logger.info("📱 Telegram notifications disabled")
                return
            
            if not self.bot_token or not self.chat_id:
                logger.error("❌ Telegram bot_token or chat_id not configured")
                self.enabled = False
                return
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test connection
            await self._test_connection()
            
            logger.info("✅ Telegram notifier initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Telegram: {e}")
            self.enabled = False
    
    async def _test_connection(self):
        """Test Telegram bot connection"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    bot_info = await response.json()
                    logger.info(f"✅ Connected to Telegram bot: {bot_info['result']['username']}")
                else:
                    raise Exception(f"HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Telegram connection test failed: {e}")
            raise
    
    async def send_startup_message(self):
        """Send bot startup notification"""
        if not self.enabled:
            return
        
        try:
            message = self.templates['startup']
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Error sending startup message: {e}")
    
    async def send_trade_signal(self, signal: dict):
        """
        Send trade signal notification
        
        Args:
            signal: Enhanced signal dict with all trade parameters
        """
        if not self.enabled:
            return
        
        try:
            # Format signal data
            side_emoji = "🟢" if signal['side'] == 'long' else "🔴"
            side_text = "LONG" if signal['side'] == 'long' else "SHORT"
            quality_emoji = self._get_quality_emoji(signal['quality'])
            
            # Calculate percentage changes for TPs and SL
            entry = signal['entry_price']
            tp1_pct = ((signal['take_profit_1'] - entry) / entry) * 100
            tp2_pct = ((signal['take_profit_2'] - entry) / entry) * 100
            tp3_pct = ((signal['take_profit_3'] - entry) / entry) * 100
            
            if signal['side'] == 'short':
                tp1_pct = -tp1_pct
                tp2_pct = -tp2_pct
                tp3_pct = -tp3_pct
            
            # Format message
            message = self.templates['signal'].format(
                quality=f"{quality_emoji} {signal['quality']}",
                pair=signal['pair'],
                side_emoji=side_emoji,
                side_text=side_text,
                entry_price=signal['entry_price'],
                tp1=signal['take_profit_1'],
                tp1_pct=tp1_pct,
                tp2=signal['take_profit_2'],
                tp2_pct=tp2_pct,
                tp3=signal['take_profit_3'],
                tp3_pct=tp3_pct,
                sl=signal['stop_loss'],
                sl_pct=signal['sl_percent'],
                leverage=signal['recommended_leverage'],
                time_limit=signal['time_limit'],
                score=signal['quality_score'],
                volume=signal['metrics'].get('volume_ratio', 1.0),
                momentum=signal['metrics'].get('momentum', 0.5),
                volatility=signal['metrics'].get('volatility', 0.02),
                timestamp=signal['timestamp'].strftime('%H:%M:%S UTC')
            )
            
            await self._send_message(message)
            
            logger.info(f"📱 Telegram signal sent for {signal['pair']} ({signal['quality']})")
            
        except Exception as e:
            logger.error(f"❌ Error sending trade signal: {e}")
    
    async def send_trade_update(self, signal: dict, current_price: float, 
                               pnl_percent: float, pnl_usd: float, 
                               update_message: str):
        """Send trade update notification"""
        if not self.enabled:
            return
        
        try:
            side_text = "LONG" if signal['side'] == 'long' else "SHORT"
            
            message = self.templates['update'].format(
                pair=signal['pair'],
                current_price=current_price,
                pnl=pnl_percent,
                pnl_usd=pnl_usd,
                update_message=update_message,
                timestamp=datetime.now().strftime('%H:%M:%S UTC')
            )
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Error sending trade update: {e}")
    
    async def send_signal_expired(self, signal: dict):
        """Send signal expiry notification"""
        if not self.enabled:
            return
        
        try:
            side_text = "LONG" if signal['side'] == 'long' else "SHORT"
            
            message = self.templates['expired'].format(
                pair=signal['pair'],
                side_text=side_text,
                entry_price=signal['entry_price'],
                timestamp=datetime.now().strftime('%H:%M:%S UTC')
            )
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Error sending expiry notification: {e}")
    
    def _get_quality_emoji(self, quality: str) -> str:
        """Get emoji for signal quality"""
        quality_emojis = {
            'GEM': '💎',
            'GOOD': '✅',
            'NEUTRAL': '⚪'
        }
        return quality_emojis.get(quality, '❓')
    
    async def _send_message(self, message: str):
        """Send message via Telegram API"""
        if not self.enabled or not self.session:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            raise
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
