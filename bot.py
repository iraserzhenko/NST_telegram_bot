from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
import model
import warnings
warnings.filterwarnings("ignore")

URI_INFO = f"https://api.telegram.org/bot{API_TOKEN}/getFile?file_id="
URI = f"https://api.telegram.org/file/bot{API_TOKEN}/"

storage = MemoryStorage()
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=storage)


class Photos(StatesGroup):
    content_id = State()
    style_id = State()


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.reply("Этот бот умеет стилизовать изображения, для этого используйте команду /style_transfer")


@dp.message_handler(commands=['style_transfer'])
async def begin_styling(message: types.Message):
    await bot.send_message(message.chat.id, "Отправь фото, которое хочешь обработать:")
    await Photos.content_id.set()


@dp.message_handler(content_types=types.ContentType.PHOTO, state=Photos.content_id)
async def photo_content(message: types.Message, state: FSMContext):
    image = message.photo[-1]
    file_info = await bot.get_file(image.file_id)
    photo = await bot.download_file(file_info.file_path)
    async with state.proxy() as proxy:
        proxy['content'] = photo
    await Photos.next()
    await message.reply("Отправь фото, стиль которого ты хочешь перенести")


@dp.message_handler(content_types=types.ContentType.PHOTO, state=Photos.style_id)
async def send_process(message: types.Message, state: FSMContext):
    image = message.photo[-1]
    file_info = await bot.get_file(image.file_id)
    photo = await bot.download_file(file_info.file_path)
    async with state.proxy() as proxy:
        proxy['style'] = photo
        new_image = model.run_nst(proxy['style'], proxy['content'])
        await bot.send_photo(message.chat.id, photo=new_image)
    await message.reply("Вот конечный результат. Если хочешь стилизовать еще одно фото, используй команду "
                        "/style_transfer")
    await state.finish()