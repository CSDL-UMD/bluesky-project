import requests, aiohttp, asyncio
import pandas as pd
import lets_plot as lp
import urllib.parse
from pathlib import Path
from lets_plot import *
from shiny import App, Inputs, Outputs, Session, render, reactive, ui
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.routing import Mount, Route
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.routing import Mount
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from murkysky_api import router as murkysky_router

app_static = StaticFiles(directory="/home/ubuntu/bluesky-project/")
min_date = "2023-12-08"
current_date = date.today()
previous_date = current_date - timedelta(days=2)
js_file = Path(__file__).parent
icon = ui.HTML(
    '<svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="13px" height="13px" viewBox="0 0 50 50">'
    '<path d="M25,2C12.297,2,2,12.297,2,25s10.297,23,23,23s23-10.297,23-23S37.703,2,25,2z M25,11c1.657,0,3,1.343,3,3s-1.343,3-3,3 s-3-1.343-3-3S23.343,11,25,11z M29,38h-2h-4h-2v-2h2V23h-2v-2h2h4v2v13h2V38z"></path>'
    '</svg>'
)
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.script(src="scroll.js")
    ),
    ui.tags.div(
        ui.tags.style(
            """
            @import url('https://fonts.googleapis.com/css2?family=Jost:wght@500&display=swap');
            body {
                font-family: Jost, sans-serif;
                box-sizing: border-box;
            }
            
            h2{
                margin-top: 20px;
            }
            
            #daterange{disabled: True}
            .checkbox-inline, .radio-inline {margin-right: 33.8px;}
            #time-label {margin-bottom: 16.4px;}
            @media screen and (max-width: 1024px) {
                .main-layout {
                    flex-direction: column !important;
                }

                .controls-section, 
                .visualization-section {
                    width: 100% !important;
                    margin-left: 0 !important;
                }
            }

            @media screen and (max-width: 768px) {
                .news-domains {
                    flex-direction: column !important;
                }

                .news-domains > div {
                    width: 100% !important;
                    margin-bottom: 1rem;
                }
            }
            """
        )
    ),
    ui.tags.div(
        ui.panel_title("MurkySky 🌩️  ☀️"),
        ui.tags.div(
            ui.output_ui("about"),
            ui.output_ui("test"),
            ui.input_action_button(
                "btn",
                "Stats",
                onclick="scrollToTopNews()",
                style="background: none; color: black; border: none; margin-bottom: 10px; padding: 10px; font-size: 23px; text-decoration: none; cursor: pointer; margin-top: 24px; margin-right: 10px;"
            ),
            ui.output_ui("api"),
            style="display: flex; align-items: center; margin-top: -1%;"
        ),
        style="display: flex; justify-content: space-between;"
    ),
    ui.tags.div(
        ui.tags.div(
            ui.tags.div(
                ui.tags.div(
                    ui.output_ui("head_text"),
                    style='font-size: 21.4px;'
                ),
                ui.tags.span("How much unreliable news has been posted on Bluesky in the last 7 days? To answer this question, we track all posts with links to news sites in real time. From this data, we created an index reflecting how much of that content comes from news sources that are considered unreliable"),
                ui.tooltip(
                    ui.HTML(f"<sup>{icon}</sup>."),
                    ui.HTML("As determined by expert journalists; more info <a href='/murkysky/about' target='_blank'>here</a>."),
                    placement="right",
                    id="card_tooltip",
                ),
                style='border: 2px solid #ccc; border-radius: 10px; padding: 17px;'
            ),
            ui.tags.div(style="margin: 15px 0;"),
            ui.tags.div( 
                ui.tags.div(
                    ui.tags.h3("Visualization controls"),
                    style = 'font-size = 24px; margin-bottom: 6%; text-align: center;'
                ),
                ui.tags.div(
                    [
                        ui.tags.label("Frequency: ", style="font-size: 16px; margin-right: 22.5%; margin-bottom: 15px;"),
                        ui.input_select(
                            "dataset",
                            None,
                            {'Hour': 'Hourly', 'Day': 'Daily'}
                        )
                    ],
                    style="display: flex; align-items: center; padding-right: 10px"
                ),
                ui.tags.div(
                    ui.tags.div(
                        [
                            ui.input_radio_buttons(
                                "time",
                                "Display data from: ",
                                choices={
                                    "seven": "Last 7 days",
                                    "thirty": "Last 30 days",
                                    "all": "All the data",
                                    "custom": "Custom range"
                                },
                            ),
                            ui.tags.div(
                                ui.input_date_range(
                                    "daterange",
                                    None,
                                    start="2023-12-08",
                                    min="2023-12-08",
                                    width= '13.3rem',
                                ),
                                style="margin-bottom: -28%; padding: 10px; display: flex; align-items: center; justify-content: flex-end;"
                            ),
                        ],
                        style="display: flex; align-items: center; gap: 10px; justify-content: space-between; width: 100%;"
                    ),
                ),
                ui.input_radio_buttons(
                    "value",
                    "Show y-axis as: ",
                    choices={"relative": "Relative values",
                     "absolute": "Absolute values"},
                    inline=True,
                    selected="relative"
                ),
                style='border: 2px solid #ccc; border-radius: 10px; padding: 20px;'
            ),
            style="width: 30%;",
        ),
        ui.div(
            ui.tags.div(
                ui.output_ui("overlayText"),
            ),
            ui.output_ui("dimensions"),
            ui.div(
                ui.output_ui("letsplot"),
            ),
            ui.div(
                ui.output_ui("secondOverlayText"),
            ),
            style="flex-grow: 1; display: flex; flex-direction: column; margin-left: 3%",
        ),
        style="display: flex;", class_="vis-container"
    ),
    ui.tags.div(
        ui.tags.div(
            ui.tags.h3("Stats", id="top_news"),
            style='font-size: 24px; margin: 1% 0 0; text-align: center;'
        ),
            ui.tags.div(
                ui.tags.label("Show stats for the last: ", style="font-size: 16px; margin: -1.1% 2.1% 0 0.25%;"),
                ui.input_radio_buttons(
                    "stats",
                    None,
                    choices={"week": "7 days", "month": "30 days"},
                    inline=True,
                    selected="week"
                ),
            style="display: flex; justify-content: center; align-items: center; border-radius: 10px; margin-top: 0.5%;"
        ),
        ui.tags.div(
            ui.tags.h4("Top News Stories"),
            ui.tags.p(
                "News rating provided by NewsGuard ",
                ui.tags.a("(methodology)", href="https://www.newsguardtech.com/ratings/rating-process-criteria/", target="_blank"),
                style="font-size: 14.5px; color: gray; margin-bottom: 15px; text-align: center;"
            ),
            style='font-size: 24px; margin: 10px 0 7px; text-align: center;'
        ),
        ui.div(
            ui.div(
               # ui.tags.h5("Reliable", style="text-decoration: underline; text-align: center; margin-bottom: 10px;"),
                ui.output_data_frame("url_df_reliable"),
                style='flex: 1; padding: 14px; overflow: auto; display: flex; flex-direction: column; align-items: center; justify-content: center;'
            ),
            ui.div(
                # ui.tags.h5("Unreliable", style="text-decoration: underline; text-align: center; margin-bottom: 10px;"),
                ui.output_data_frame("url_df_unreliable"),
                style='flex: 1; padding: 14px; overflow: auto; display: flex; flex-direction: column; align-items: center; justify-content: center;'
            ),
            style='display: flex; justify-content: space-between; align-items: stretch; border: 2px solid #ccc; border-radius: 10px; padding: 30px; gap: 20px;'
        ),
        #ui.tags.div(
        #    ui.tags.h4("Most Shared News Domains"),
        #    ui.tags.p(
        #        "News rating provided by NewsGuard ",
        #        ui.tags.a("(methodology)", href="https://www.newsguardtech.com/ratings/rating-process-criteria/", target="_blank"),
        #        style="font-size: 14.5px; color: gray; margin-bottom: 15px; text-align: center;"
        #    ),
        #    style='font-size: 24px; margin-top: 16px; text-align: center;'
        #),
        #ui.div(
        #    ui.div(
        #        ui.tags.h5("Reliable", style="text-decoration: underline; text-align: center; margin-bottom: 10px;"),
        #        ui.output_data_frame("domain_df_reliable"),
        #        style='flex: 1; padding: 10px; display: flex; flex-direction: column; align-items: center; width: 100%; overflow-x: auto;'
        #    ),
        #    ui.div(
        #        ui.tags.h5("Unreliable", style="text-decoration: underline; text-align: center; margin-bottom: 10px;"),
        #        ui.output_data_frame("domain_df_unreliable"),
        #        style='flex: 1; padding: 10px; display: flex; flex-direction: column; align-items: center; width: 100%; overflow-x: auto;'
        #    ),
        #    style='display: flex; flex-wrap: wrap; justify-content: space-between; align-items: stretch; border: 2px solid #ccc; border-radius: 10px; padding: 30px;'
        #),
    ),
    ui.tags.div(
        ui.tags.div(
            ui.HTML("<img src='https://umd-brand.transforms.svdcdn.com/production/uploads/images/informal-seal.png?w=512&h=512&auto=compress%2Cformat&fit=crop&dm=1656362660&s=f147c43be06ac2a530c41260819e63a1' alt='University of Maryland Logo' style='height: 4%; width: 4%; margin-right: 20px; margin-bottom: 7px;'></img>"),
            ui.HTML("<span>Copyright © 2024. Computational Social Dynamics Lab. </span>"),
            style="display: flex; align-items: center; justify-content: center; gap: 0; margin-top: 5px;"
        )
    )
)

def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    def dimensions():
        return ui.tags.script("""
        $(document).ready(function() {
            var plotContainer = document.getElementById('letsplot');
            const element = document.querySelector('.vis-container');
            if (plotContainer && element) {
                var width = plotContainer.offsetWidth;
                Shiny.setInputValue('letsplot_width', width);
                var height = element.offsetHeight;
                Shiny.setInputValue('letsplot_height', height);
            }
        });
        """)

    @reactive.Calc
    def plot_width():
        return input.letsplot_width()

    @reactive.Calc
    def plot_height():
        return input.letsplot_height()

    @output(id='letsplot')
    @render.ui
    def compute():
        selection = input.dataset()
        radio = input.time()
        chart = input.value()
        date = input.daterange()
        end_date = datetime.today().strftime('%Y-%m-%d')
        response = requests.get("http://10.224.109.230:3001/get_data")
        if response.status_code == 200:
            data = response.json()
            columns_as_array = list(map(list, zip(*data)))
        else:
            print("Error fetching data. Status code:", response.status_code)
        now_floored = datetime.now().replace(minute=0, second=0, microsecond=0)
        start_date7 = now_floored - timedelta(days=7)
        start_date30 = now_floored - timedelta(days=30)
        day_array = columns_as_array[0]
        day_array_dt = pd.to_datetime(day_array)
        rounded_day_array = [dt.strftime('%Y-%m-%d %H:%M') for dt in day_array_dt]
        total_messages_array = columns_as_array[1]
        total_links_array = columns_as_array[2]
        news_greater_than_60_array = columns_as_array[3]
        news_less_than_60_array = columns_as_array[4]
        relative_news = [
            news_less_than_60 / (news_less_than_60 + news_greater_than_60) if news_greater_than_60 != 0 and news_less_than_60 != 0 else 0
            for news_less_than_60, news_greater_than_60 in zip(news_less_than_60_array, news_greater_than_60_array)
        ]
        df_data = {
            'Day': rounded_day_array,
            'TotalMessages': total_messages_array, 
            'TotalLinks': total_links_array,
            'NewsGreaterThan60': news_greater_than_60_array,
            'NewsLessThan60': news_less_than_60_array,
            'RelativeNews': relative_news,
        }
        df = pd.DataFrame(df_data)
        df['Timestamp'] = df['Day'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M', errors='coerce')
        df = df[df['Timestamp'].dt.strftime("%Y-%m-%d") != '2023-12-07']
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.sort_values(by='Timestamp')
        start_date = df.loc[df['Timestamp'].notna(), 'Timestamp'].iloc[0].replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = df['Timestamp'].max()
        complete_date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        complete_df = pd.DataFrame({'Timestamp': complete_date_range})
        df['Rounded_Timestamp'] = df['Timestamp'].dt.floor('H')
        df = pd.merge(complete_df, df, left_on='Timestamp', right_on='Rounded_Timestamp', how='left')
        df = df.drop(columns=['Rounded_Timestamp'])
        df['Timestamp'] = df['Timestamp_y']
        df = df.drop(columns=['Timestamp_y'])
        
        df['Timestamp'] = df['Timestamp'].fillna(df['Timestamp_x'])
        df.loc[df['Timestamp'].isna(), df.columns != 'Timestamp'] = 0
        rows_to_delete = []

        for i in range(1, len(df)-1):
            time_diff = (df.loc[i+1, 'Timestamp'] - df.loc[i-1, 'Timestamp']).total_seconds() / 60
            if time_diff <= 110 and pd.isna(df.loc[i, 'TotalLinks']):
                rows_to_delete.append(i)
        df.drop(rows_to_delete, inplace=True)

        df['Timestamp'] = df['Timestamp'] + pd.to_timedelta(df.groupby('Timestamp').cumcount(), unit='H')
        df['Day'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        df = df.fillna(0)
        df['TotalMessagesDaily'] = df.groupby(df['Timestamp'].dt.date)['TotalMessages'].transform('sum')
        df['DateAndTotalMessages'] = df['Timestamp'].dt.strftime('%Y-%m-%d')
        df['TotalLinksDaily'] = df.groupby(df['Timestamp'].dt.date)['TotalLinks'].transform('sum')
        df['NewsGreaterThan60Daily'] = df.groupby(df['Timestamp'].dt.date)['NewsGreaterThan60'].transform('sum')
        df['NewsLessThan60Daily'] = df.groupby(df['Timestamp'].dt.date)['NewsLessThan60'].transform('sum')
        df['NumTimestampsDaily'] = df.groupby(df['Timestamp'].dt.date)['Timestamp'].transform('count')
        df['RelativeNewsDaily'] = df.groupby(df['Timestamp'].dt.date)['RelativeNews'].transform('sum') / df['NumTimestampsDaily']
        df = df.drop(columns=['NumTimestampsDaily'])
        df['Day2'] = df['Day']
        df['Day2'] = pd.to_datetime(df['Day2'], errors='coerce')
        df["Day2"] = df["Day2"].dt.strftime('%b %d %Y %H:%M')
        df['Day'] = pd.to_datetime(df['Day'], format='%Y-%m-%d %H:%M', errors='coerce')
        df['DateAndTotalMessages2'] = df["DateAndTotalMessages"]
        df['DateAndTotalMessages'] = pd.to_datetime(df['DateAndTotalMessages'])
        
        df2 = df[df['Timestamp_x'] >= start_date7].copy()
        df3 = df[df['Timestamp_x'] >= start_date30].copy()
        df4 = df[(df['Timestamp'].dt.date >= date[0]) & (df['Timestamp'].dt.date <= date[1])].copy()
        df4['Day2'] = pd.to_datetime(df4['Day2'], format="%b %d %Y %H:%M")
 
        date_difference = relativedelta(date[1], date[0])
        months_difference = date_difference.months
        days_difference = date_difference.days

        first_date_as_string = rounded_day_array[0]
        last_date_as_string = rounded_day_array[-1]
        first_date_as_datetime = pd.to_datetime(first_date_as_string)
        last_date_as_datetime = pd.to_datetime(last_date_as_string)
        all_date_difference = relativedelta(last_date_as_datetime, first_date_as_datetime)
        all_months_difference = all_date_difference.months
        tooltips = layer_tooltips().format('Day', '%b %d %Y')
        custom_tooltips = layer_tooltips().format('Day2', '%b %d %Y')
        dtooltips = layer_tooltips().format('DateAndTotalMessages', '%b %d %Y')
        tooltips_hour = layer_tooltips().format('Day2', '%b %d %Y %H:%M')
        all_mean = df.loc[df["RelativeNews"] != 0, "RelativeNews"].mean()
        seven_mean = df2.loc[df2["RelativeNews"] != 0, "RelativeNews"].mean()
        thirty_mean = df3.loc[df3["RelativeNews"] != 0, "RelativeNews"].mean()
        custom_mean = df4.loc[df4["RelativeNews"] != 0, "RelativeNews"].mean()
        weather_mapping = {
            (0.0, 0.10): "☀️",
            (0.10, 0.20): "🌤️",
            (0.20, 0.30): "⛅️",
            (0.30, 0.40): "☁️",
            (0.40, 0.50): "🌦️",
            (0.50, 0.60): "🌧️",
            (0.60, 0.70): "🌩️",
            (0.70, 0.80): "⛈️",
            (0.80, 0.90): "🌨️",
            (0.90, 1.00): "🌪️",
        }
        all_mean_emoji = next((emoji for range_, emoji in weather_mapping.items() if range_[0] <= all_mean <= range_[1]), None)
        seven_mean_emoji = next((emoji for range_, emoji in weather_mapping.items() if range_[0] <= seven_mean <= range_[1]), None)
        thirty_mean_emoji = next((emoji for range_, emoji in weather_mapping.items() if range_[0] <= thirty_mean <= range_[1]), None)
        custom_mean_emoji = next((emoji for range_, emoji in weather_mapping.items() if range_[0] <= custom_mean <= range_[1]), None)
        df2_feb_26 = df2[df2['Timestamp'].dt.day == 26]
        width, height = plot_width(), plot_height() * 1.10154905336
        if selection == 'Hour':
            if radio == "seven":
                if chart == "absolute":
                    p = (
                        lp.ggplot(df2)
                        + lp.geom_area(lp.aes(x='Day', y='NewsLessThan60'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=tooltips)
                        + lp.geom_area(lp.aes(x='Day', y='NewsGreaterThan60'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=tooltips)
                        + lp.geom_area(lp.aes(x='Day', y='TotalLinks'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=tooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                    )
                elif chart == "relative":
                    p = (
                            lp.ggplot(df2)
                            + lp.geom_area(lp.aes(x='Day', y='RelativeNews'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=tooltips)
                            + lp.ggsize(width,height)
                            + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                            + lp.xlab("Date")
                            + lp.ylab("Proportion of News Links")
                            + lp.theme(text=lp.element_text(family='Georgia'))
                            + lp.theme(panel_grid_major_x=lp.element_blank())
                            + geom_hline(yintercept=seven_mean, color="black", linetype="dashed")
                    )
                else:
                    raise ValueError(f'{chart=} is not valid.')
                p = p + lp.scale_x_datetime(format='%b %e %Y')
            if radio == "thirty":
                if chart == "absolute":
                    p = (
                        lp.ggplot(df3)
                        + lp.geom_area(lp.aes(x='Day', y='NewsLessThan60'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=tooltips)
                        + lp.geom_area(lp.aes(x='Day', y='NewsGreaterThan60'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=tooltips)
                        + lp.geom_area(lp.aes(x='Day', y='TotalLinks'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=tooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                    )
                elif chart == "relative":
                    p = (
                        lp.ggplot(df3)
                        + lp.geom_area(lp.aes(x='Day', y='RelativeNews'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=tooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Proportion of News Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_hline(yintercept=thirty_mean, color="black", linetype="dashed")
                    )
                else:
                    raise ValueError(f'{chart=} is not valid.')
                p = p + lp.scale_x_datetime(format='%b %e %Y')
            elif radio == "all":
                if chart == "absolute":
                    p = (
                        lp.ggplot(df)
                        + lp.geom_area(lp.aes(x='Day', y='NewsLessThan60'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=tooltips)
                        + lp.geom_area(lp.aes(x='Day', y='NewsGreaterThan60'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=tooltips)
                        + lp.geom_area(lp.aes(x='Day', y='TotalLinks'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=tooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + geom_vline(xintercept=pd.Timestamp("2024-06-14 23:00:00"), color="black", linetype="dotted")
                        + geom_vline(xintercept=pd.Timestamp("2024-08-29 19:30:00"), color="black", linetype="solid")
                        + geom_vline(xintercept=pd.Timestamp("2024-11-22 18:00:00"), color="black", linetype="longdash")
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                    )
                elif chart == "relative":
                    p = (
                        lp.ggplot(df)
                        + lp.geom_area(lp.aes(x='Day', y='RelativeNews'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=tooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Proportion of News Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + geom_vline(xintercept=pd.Timestamp("2024-11-22 18:00:00"), color="black", linetype="longdash")
                        + geom_vline(xintercept=pd.Timestamp("2024-08-29 19:30:00"), color="black", linetype="solid")
                        + geom_vline(xintercept=pd.Timestamp("2024-06-14 23:00:00"), color="black", linetype="dotted")
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_hline(yintercept=all_mean, color="black", linetype="dashed")
                    )
                else:
                    raise ValueError(f'{chart=} is not valid.')    
                if all_months_difference >= 3:
                    p = p + lp.scale_x_datetime(format='%b %Y')
                else:
                    p = p + lp.scale_x_datetime(format='%b %e %Y')
            elif radio == "custom":
                if months_difference != 0 or days_difference > 5:
                    tool = custom_tooltips
                else:
                    tool = tooltips_hour 
                data_collection_datetime = pd.to_datetime('2024-06-14T23:00:00.000000000')
                data_collection_datetime2 = pd.to_datetime('2024-08-29T19:00:00.000000000')
                data_collection_datetime3 = pd.to_datetime('2024-11-22T19:00:00.000000000') 
                if chart == "absolute":
                    p = (
                        lp.ggplot(df4)
                        + lp.geom_area(lp.aes(x='Day2', y='NewsLessThan60'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=tool)
                        + lp.geom_area(lp.aes(x='Day2', y='NewsGreaterThan60'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=tool)
                        + lp.geom_area(lp.aes(x='Day2', y='TotalLinks'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=tool)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                    )
                elif chart == "relative":
                    p = (
                        lp.ggplot(df4)
                        + lp.geom_area(lp.aes(x='Day2', y='RelativeNews'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=tool)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Proportion of News Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_hline(yintercept=custom_mean, color="black", linetype="dashed")
                    )
                else:
                    raise ValueError(f'{chart=} is not valid.')
                if not df4[df4['Timestamp_x'] == data_collection_datetime3].empty:
                    p = p + geom_vline(xintercept=pd.Timestamp("2024-11-22 18:00:00"), color="black", linetype="longdash")
                if not df4[df4['Timestamp_x'] == data_collection_datetime2].empty:
                    p = p + geom_vline(xintercept=pd.Timestamp("2024-08-29 19:30:00"), color="black", linetype="solid")
                if not df4[df4['Timestamp_x'] == data_collection_datetime].empty:
                    p = p + geom_vline(xintercept=pd.Timestamp("2024-06-14 23:00:00"), color="black", linetype="dotted")
            if all_months_difference >= 3:
                p = p + lp.scale_x_datetime(format='%b %Y')
            else:
                p = p + lp.scale_x_datetime(format='%b %e %Y')
        elif selection == 'Day':
            if radio == "thirty":
                if chart == "absolute":
                    p = (
                        lp.ggplot(df3)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsLessThan60Daily'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsGreaterThan60Daily'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='TotalLinksDaily'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                    )
                elif chart == "relative":
                    p = (
                        lp.ggplot(df3)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='RelativeNewsDaily'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Proportion of News Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_hline(yintercept=thirty_mean, color="black", linetype="dashed")
                    )
                else:
                    raise ValueError(f'{chart=} is not valid.')
                p = p + lp.scale_x_datetime(format='%b %e %Y')
            elif radio == "seven":
                if chart == "absolute":
                    p = (
                        lp.ggplot(df2)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsLessThan60Daily'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsGreaterThan60Daily'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='TotalLinksDaily'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                    )
                elif chart == "relative":
                    p = (
                        lp.ggplot(df2)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='RelativeNewsDaily'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Proportion of News Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_hline(yintercept=seven_mean, color="black", linetype="dashed")
                    )
                else:
                    raise ValueError(f'{chart=} is not valid.')
                p = p + lp.scale_x_datetime(format='%b %e %Y')
            elif radio == "all":
                if chart == "absolute":
                    p = (
                        lp.ggplot(df)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsLessThan60Daily'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsGreaterThan60Daily'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='TotalLinksDaily'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_vline(xintercept=pd.Timestamp("2024-11-22"), color="black", linetype="longdash")
                        + geom_vline(xintercept=pd.Timestamp("2024-08-29"), color="black", linetype="solid")
                        + geom_vline(xintercept=pd.Timestamp("2024-06-14"), color="black", linetype="dotted")
                    )
                elif chart == "relative":
                    p = (
                        lp.ggplot(df)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='RelativeNewsDaily'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Proportion of News Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_hline(yintercept=all_mean, color="black", linetype="dashed")
                        + geom_vline(xintercept=pd.Timestamp("2024-11-22"), color="black", linetype="longdash")
                        + geom_vline(xintercept=pd.Timestamp("2024-08-29"), color="black", linetype="solid")
                        + geom_vline(xintercept=pd.Timestamp("2024-06-14"), color="black", linetype="dotted")
                    )
                else:
                    raise ValueError(f'{chart=} is not valid.')
                if all_months_difference >= 3:
                    p = p + lp.scale_x_datetime(format='%b %Y')
                else:
                    p = p + lp.scale_x_datetime(format='%b %e %Y')
            elif radio == "custom":
                data_collection_datetime3 = pd.to_datetime('November 22 2024', format="%B %d %Y")
                data_collection_datetime = pd.to_datetime('June 14 2024', format="%B %d %Y")
                data_collection_datetime2 = pd.to_datetime('August 29 2024', format="%B %d %Y")
                if chart == "absolute":
                    p = (
                        lp.ggplot(df4)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsLessThan60Daily'), size=1, color='#FF0000', fill='#FF0000', alpha=0.30, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='NewsGreaterThan60Daily'), size=1, color='#00008B', fill='#00008B', alpha=0.25, position="identity", tooltips=dtooltips)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='TotalLinksDaily'), size=1, color='#83F28F', fill='#83F28F', alpha=0.40, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Number of Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                    )
                elif chart == "relative":
                    p = (
                        lp.ggplot(df4)
                        + lp.geom_area(lp.aes(x='DateAndTotalMessages', y='RelativeNewsDaily'), size=1, color='#702963', fill='#702963', alpha=0.45, position="identity", tooltips=dtooltips)
                        + lp.ggsize(width,height)
                        + lp.theme(axis_text_x=lp.element_text(angle=360, hjust=1))
                        + lp.xlab("Date")
                        + lp.ylab("Proportion of News Links")
                        + lp.theme(text=lp.element_text(family='Georgia'))
                        + lp.theme(panel_grid_major_x=lp.element_blank())
                        + geom_hline(yintercept=custom_mean, color="black", linetype="dashed")
                    )
                else:
                    raise ValueError(f'{selection=} is not valid.')
                if not df4[df4['Timestamp_x'] == data_collection_datetime3].empty:
                    p = p + geom_vline(xintercept=pd.Timestamp("2024-11-22"), color="black", linetype="longdash")
                if not df4[df4['Timestamp_x'] == data_collection_datetime2].empty:
                    p = p + geom_vline(xintercept=pd.Timestamp("2024-08-29"), color="black", linetype="solid")
                if not df4[df4['Timestamp_x'] == data_collection_datetime].empty:
                    p = p + geom_vline(xintercept=pd.Timestamp("2024-06-14"), color="black", linetype="dotted")

                if all_months_difference >= 3:
                    p = p + lp.scale_x_datetime(format='%b %Y')
                else:
                    p = p + lp.scale_x_datetime(format='%b %d %Y')

        @output(id='head_text')
        @render.ui
        def head_text():
            radio = input.time()
            emoji_val = seven_mean
            condition = seven_mean_emoji
            weather_mapping2 = {
                (0.0, 0.10): "Sunny",
                (0.10, 0.20): "Partly Cloudy",
                (0.20, 0.30): "Mostly Cloudy",
                (0.30, 0.40): "Cloudy",
                (0.40, 0.50): "Showers",
                (0.50, 0.60): "Rainy",
                (0.60, 0.70): "Thunderstorm",
                (0.70, 0.80): "Heavy Thunderstorm",
                (0.80, 0.90): "Snowy",
                (0.90, 1.00): "Tornado",
            }
            condition2 = next((value for key, value in weather_mapping2.items() if key[0] < emoji_val <= key[1]), None)
            rounded_time = df['Day'].iloc[-1] + pd.Timedelta(minutes=15)
            formatted_time = rounded_time.floor('30min').strftime('%b %d, %Y at %I:%M %p')
            if condition:
                return ui.tags.div(
                    ui.HTML('<span style="font-size:19.7px;">Current Weather on Bluesky:</span><br>'),
                    ui.tags.span(f"{condition2} ", style="font-size:19.7px;"),
                    ui.tags.span(f"{condition} ", style="font-size:28px;"),
                    ui.tags.span(f"({emoji_val * 100:.0f}% unreliable news)", style="font-size:19.7px;"),
                    ui.tags.p(
                        f"(Last updated: {formatted_time} GMT)",
                        style="font-size: 14.5px; color: gray; margin-top: 5px;margin-bottom: 15px;"
                    )
                )
            else:
                return ui.tags.p("Invalid emoji value")
        
        @output(id='test')
        @render.ui
        def about():
            return ui.tags.div(
                ui.HTML('<a href="/murkysky/about" target="_blank" style="color: black; text-decoration: none;">About</a>'),
                style="margin-bottom: 10px; margin-top: 24px; margin-right: 10px; font-size: 23px"
            )
      
        @output(id='about')
        @render.ui
        def about():
            return ui.tags.div(
                ui.HTML('<a href="/murkysky/about" target="_blank" style="color: black; text-decoration: none;">About</a>'),
                style="margin-bottom: 10px; margin-top: 24px; margin-right: 10px; font-size: 23px"
            )

        @output(id='api')
        @render.ui
        def about():
            return ui.tags.div(
                ui.HTML('<a href="https://rapidapi.com/csdl-umd-csdl-umd-default/api/murkysky-api" target="_blank" style="color: black; text-decoration: none;">API</a>'),
                style="margin-bottom: 10px; margin-top: 24px; margin-right: 10px; font-size: 23px"
            )

        #@render.data_frame
        #@reactive.event(input.stats, fetch_data.result)
        #def domain_df_reliable():
        #    radio = input.stats()
        #    if radio == "week":
        #        url_table = pd.DataFrame(fetch_data.result().get('domain_week_high'), columns=['Domain', 'Total Count'])
        #        return render.DataGrid(url_table)
        #    elif radio == "month":
        #        url_table = pd.DataFrame(fetch_data.result().get('domain_month_high'), columns=['Domain', 'Total Count'])
        #        return render.DataGrid(url_table)

        #@render.data_frame
        #@reactive.event(input.stats, fetch_data.result)
        #def domain_df_unreliable():
        #    radio = input.stats()
        #    if radio == "week":
        #        domain_table = pd.DataFrame(fetch_data.result().get('domain_week_low'), columns=['Domain', 'Total Count'])
        #        return render.DataGrid(domain_table)
        #    elif radio == "month":
        #        domain_table = pd.DataFrame(fetch_data.result().get('domain_month_low'), columns=['Domain', 'Total Count'])
        #        return render.DataGrid(domain_table)

        @render.data_frame
        @reactive.event(input.stats, fetch_data.result)
        def url_df_reliable():
            radio = input.stats()
            if radio == "week":
                url_table = pd.DataFrame(fetch_data.result().get('url_week_high'), columns=['URL', 'Total Count'])
                return render.DataGrid(url_table, width='675px', height='400px')
            elif radio == "month":
                url_table = pd.DataFrame(fetch_data.result().get('url_month_high'), columns=['URL', 'Total Count'])
                return render.DataGrid(url_table, width='675px', height='400px')

        @render.data_frame
        @reactive.event(input.stats, fetch_data.result)
        def url_df_unreliable():
            radio = input.stats()
            if radio == "week":
                url_table = pd.DataFrame(fetch_data.result().get('url_week_low'), columns=['URL', 'Total Count'])
                return render.DataGrid(url_table, width='675px', height='400px')
            elif radio == "month":
                url_table = pd.DataFrame(fetch_data.result().get('url_month_low'), columns=['URL', 'Total Count'])
                return render.DataGrid(url_table, width='675px', height='400px')

        @output(id='secondOverlayText')
        @render.ui
        @reactive.event(input.value)
        def secondOverlayText():
            note = ""
            chart = input.value()
            radio = input.time()
            selection = input.dataset()
            if radio == "seven":
                df5 = df2
            elif radio == "thirty":
                df5 = df3
            elif radio == "all":
                df5 = df
            elif radio == "custom":
                df5 = df4
                
            if selection == "Hour":
                date_format = "%b %d %Y %H:%M"
            elif selection == "Day":
                date_format = "%b %d %Y"
                
            df5['Day'] = pd.to_datetime(df5['Day'], format=date_format, errors='coerce')
            
            markers = [
                {
                    'date': 'June 14 2024',
                    'hour_date': '2024-06-14T23:00:00.000000000',
                    'message': 'Added reposts and likes from',
                    'hour_message': 'Added reposts and likes from',
                    'hour_time': '23:00 GMT',
                    'line_style': 'border-top: 2px dotted gray'
                },
                {
                    'date': 'August 29 2024',
                    'hour_date': '2024-08-29T19:00:00.000000000',
                    'message': 'Outage started from',
                    'hour_message': 'Outage started from',
                    'hour_time': '19:30 GMT',
                    'line_style': 'border-top: 2px solid gray'
                },
                {
                    'date': 'November 22 2024',
                    'hour_date': '2024-11-22T17:00:00.000000000',
                    'message': 'Outage ended on',
                    'hour_message': 'Outage ended on',
                    'hour_time': '18:00 GMT',
                    'line_style': 'border-top: 2px dashed gray'
                }
            ]
            
            if selection == "Day":
                for marker in markers:
                    date = pd.to_datetime(marker['date'], format="%B %d %Y") 
                    if not df5[df5['Timestamp_x'] == date].empty:
                        note += f"""
                        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 10px; margin-bottom: 5px; margin-right: 10px;">
                            <div style="width: 35px; {marker['line_style']}"></div>
                            <div>{marker['message']} {marker['date']}</div>
                        </div>
                        """
                return ui.tags.div(
                    ui.HTML(note),
                    style="""
                        font-size: 12.5px;
                        color: gray;
                        margin: 5px;
                        max-width: 100%;
                        padding: 1px;
                        border-radius: 5px;
                        display: flex;
                        justify-content: flex-end;
                        align-self: flex-end;
                        text-align: right;
                    """
                )
            else:
                for marker in markers:
                    date = pd.to_datetime(marker['hour_date'])
                    if date in df5['Timestamp_x'].values:
                        note += f"""
                        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 10px; margin-bottom: 5px; margin-right: 10px;">
                            <div style="width: 30px; {marker['line_style']}"></div>
                            <div>{marker['hour_message']} {marker['date']} {marker['hour_time']}</div>
                        </div>
                        """
            return ui.tags.div(
                ui.HTML(note),
                style="""
                    font-size: 11.5px;
                    color: gray;
                    margin: 5px;
                    max-width: 100%;
                    padding: 1px;
                    border-radius: 5px;
                    display: flex;
                    justify-content: flex-end;
                    align-self: flex-end;
                    text-align: right;
                """
            )

        @output(id='overlayText')
        @render.ui
        @reactive.event(input.value)
        def overlay_text():
            chart = input.value()
            radio = input.time()
            selection = input.dataset()
            if radio == "seven":
                emoji = seven_mean_emoji
                mean = seven_mean
            elif radio == "thirty":
                emoji = thirty_mean_emoji
                mean = thirty_mean
            elif radio == "all":
                emoji = all_mean_emoji
                mean = all_mean
            elif radio == "custom":
                emoji = custom_mean_emoji
                mean = custom_mean
            if chart == "relative":
                return (
                    ui.tags.div(
                        ui.tags.div(
                            ui.tags.p(
                                ui.tags.span(style="color: rgba(112, 41, 99, 0.75); background-color: rgba(112, 41, 99, 0.75); padding: 0px 25px; margin-right: 10px; font-size: 12px;", _class="legend-box"),
                                "Ratio of Unreliable Links",
                                ui.tags.span(style="margin-right: 20px;"),
                                ui.tags.span(style="padding: 0px 25px 0px 20px; margin-right: 10px; border-bottom: 2px dashed #000; text-align: center; height: 10px; position: relative; top: -10px;"),
                                f"\nAverage Ratio of Unreliable Links ({mean * 100:.0f}% {emoji})"
                            ),
                            style="display: flex; justify-content: flex-end; flex-wrap: wrap;"
                        ),
                    style="display: flex; flex-direction: column; justify-content: flex-end; margin-bottom: -10px;"
                )
            )
            elif chart == "absolute":
                 return (
                    ui.tags.div(
                        ui.tags.div(
                            ui.tags.p(
                                ui.tags.span(style="color: rgba(208, 250, 211, 1); background-color: rgba(208, 250, 211, 1); padding: 0px 25px; margin-right: 10px; font-size: 12px;", _class="legend-box"),
                                "Total Links",
                                ui.tags.span(style="margin-right: 20px;"),
                                ui.tags.span(style="color: rgba(56, 100, 148, 0.49); background-color: rgba(56, 100, 148, 0.49); padding: 0px 25px; margin-right: 10px; font-size: 12px;", _class="legend-box"),
                                "Reliable Links",
                                ui.tags.span(style="margin-right: 20px;"),
                                ui.tags.span(style="color: rgba(184, 116, 92, 0.65); background-color: rgba(184, 116, 92, 0.65); padding: 0px 25px; margin-right: 10px; font-size: 12px;", _class="legend-box"),
                                "Unreliable Links",
                            ),
                            style="display: flex; justify-content: flex-end; margin-bottom: -10px;"
                        ),
                    style="display: flex; flex-direction: column; justify-content: flex-end; margin-bottom: -10px;"
                )
            )
        phtml = lp._kbridge._generate_static_html_page(p.as_dict(), iframe=True)
        return ui.HTML(phtml)
   
    async def fetch_data_async(type_of_url, interval, score_threshold_min, score_threshold_max):
        url = "http://10.224.109.230:3001/get_links"
        params = {
            'type_of_url': type_of_url,
            'interval': interval,
            'score_threshold_min': score_threshold_min,
            'score_threshold_max': score_threshold_max
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return []
        except aiohttp.ClientError as e:
            print(f"Error occurred: {e}")
            return []

    @reactive.extended_task
    async def fetch_data():
        results = await asyncio.gather(
            fetch_data_async('url', 'week', 60, 100),
            fetch_data_async('url', 'week', 0, 59),
            fetch_data_async('url', 'month', 60, 100),
            fetch_data_async('url', 'month', 0, 59)
        )
        #fetch_data_async('domain', 'week', 60, 100),
        #fetch_data_async('domain', 'week', 0, 59),
        #fetch_data_async('domain', 'month', 60, 100),
        #fetch_data_async('domain', 'month', 0, 59),
        return {
            'url_week_high': results[0],
            'url_week_low': results[1],
            'url_month_high': results[2],
            'url_month_low': results[3]
        }
        #'domain_week_high': results[0],
        #'domain_week_low': results[2],
        #'domain_month_high': results[4],
        #'domain_month_low': results[6],
    @reactive.Effect
    @reactive.event(input.dataset)
    async def _():
        selection = input.dataset()
        radio = input.time()
        if selection == "Hour":
            ui.update_radio_buttons("time", selected="seven")
        elif selection == "Day":
            ui.update_radio_buttons("time", selected="thirty")
        fetch_data()


app_shiny = App(app_ui, server, static_assets=js_file)
async def about(request):
    with open("./static/about.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

app = FastAPI()
app.add_route("/about", about)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(murkysky_router)
app.mount("/", app=app_shiny)
app.add_route("/about", about)

