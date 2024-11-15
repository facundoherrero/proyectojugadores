import streamlit as st
import torch
import torchvision
from PIL import Image
from utils import model_interp,vae_loaded,affect



import random


st.title("Bienvenido! Elija un filtro para comenzar a generar jugadores de futbol ⚽")

import pandas as pd
data = pd.read_csv("data/filtered_data_fix2_withimages2.csv")

# Create dropdown for 'a', 'b', or 'c'
option = st.selectbox('Elija un filtro:', ['Por Nacionalidad', 'Por Posicion', 'Por club'])

# Create dependent dropdown based on the first selection
if option == 'Por Nacionalidad':
    sub_option = st.selectbox('Elija un pais:', ['Albania', 'Algeria', 'Angola', 'Argentina', 'Australia', 'Austria', 'Belgium', 'Benin', 'Bolivia', 'Bosnia Herzegovina', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Cameroon', 'Canada', 'Cape Verde', 'Chile', 'China PR', 'Colombia', 'Congo', 'Costa Rica', 'Croatia', 'Curacao', 'Czech Republic', 'DR Congo', 'Denmark', 'Ecuador', 'Egypt', 'England', 'Estonia', 'FYR Macedonia', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guinea', 'Honduras', 'Hungary', 'Iceland', 'Iran', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Korea Republic', 'Kosovo', 'Madagascar', 'Mali', 'Mexico', 'Montenegro', 'Morocco', 'Netherlands', 'New Zealand', 'Nigeria', 'Northern Ireland', 'Norway', 'Panama', 'Paraguay', 'Peru', 'Poland', 'Portugal', 'Republic of Ireland', 'Romania', 'Russia', 'Saudi Arabia', 'Scotland', 'Senegal', 'Serbia', 'Slovakia', 'Slovenia', 'South Africa', 'Spain', 'Sweden', 'Switzerland', 'Tunisia', 'Turkey', 'Ukraine', 'United States', 'Uruguay', 'Venezuela', 'Wales', 'Zimbabwe'])
elif option == 'Por Posicion':
    sub_option = st.selectbox('Elija una posicion:', ['GK', 'DEF', 'MID', 'FWD'])
elif option == 'Por club':
    sub_option = st.selectbox('Elija un club:', [' SSV Jahn Regensburg', '1. FC Heidenheim 1846', '1. FC Kaiserslautern', '1. FC Köln', '1. FC Magdeburg', '1. FC Nürnberg', '1. FC Union Berlin', '1. FSV Mainz 05', 'AC Ajaccio', 'AC Horsens', 'AD Alcorcón', 'ADO Den Haag', 'AEK Athens', 'AFC Wimbledon', 'AIK', 'AJ Auxerre', 'AS Monaco', 'AS Nancy Lorraine', 'AS Saint-Étienne', 'AZ Alkmaar', 'Aalborg BK', 'Aarhus GF', 'Aberdeen', 'Accrington Stanley', 'Adelaide United', 'Ajax', 'Akhisar Belediyespor', 'Al Ahli', 'Al Batin', 'Al Faisaly', 'Al Fateh', 'Al Fayha', 'Al Hazem', 'Al Hilal', 'Al Ittihad', 'Al Nassr', 'Al Qadisiyah', 'Al Raed', 'Al Shabab', 'Al Taawoun', 'Al Wehda', 'Alanyaspor', 'Albacete BP', 'Alianza Petrolera', 'Amiens SC', 'América de Cali', 'Angers SCO', 'Antalyaspor', 'Arka Gdynia', 'Arsenal', 'Ascoli', 'Aston Villa', 'Atalanta', 'Athletic Club de Bilbao', 'Atiker Konyaspor', 'Atlanta United', 'Atlético Bucaramanga', 'Atlético Huila', 'Atlético Madrid', 'Atlético Nacional', 'Atlético Tucumán', 'Audax Italiano', 'BB Erzurumspor', 'BK Häcken', 'BSC Young Boys', 'Barnsley', 'Bayer 04 Leverkusen', 'Beijing Renhe FC', 'Beijing Sinobo Guoan FC', 'Belgrano de Córdoba', 'Benevento', 'Beşiktaş JK', 'Birmingham City', 'Blackburn Rovers', 'Blackpool', 'Boavista FC', 'Boca Juniors', 'Bohemian FC', 'Bologna', 'Bolton Wanderers', 'Borussia Dortmund', 'Borussia Mönchengladbach', 'Bournemouth', 'Bradford City', 'Bray Wanderers', 'Brentford', 'Brescia', 'Brighton & Hove Albion', 'Brisbane Roar', 'Bristol City', 'Bristol Rovers', 'Brøndby IF', 'Burnley', 'Bursaspor', 'Burton Albion', 'Bury', 'CA Osasuna', 'CD Antofagasta', 'CD Aves', 'CD Everton de Viña del Mar', 'CD Feirense', 'CD Huachipato', 'CD Leganés', 'CD Lugo', 'CD Nacional', 'CD Numancia', "CD O'Higgins", 'CD Palestino', 'CD Tenerife', 'CD Tondela', 'CD Universidad de Concepción', 'CF Rayo Majadahonda', 'CF Reus Deportiu', 'Cagliari', 'Cambridge United', 'Cardiff City', 'Carlisle United', 'Carpi', 'Celtic', 'Central Coast Mariners', 'Cerezo Osaka', 'Chamois Niortais Football Club', 'Changchun Yatai FC', 'Charlton Athletic', 'Chelsea', 'Cheltenham Town', 'Chicago Fire', 'Chievo Verona', 'Chongqing Dangdai Lifan FC SWM Team', 'Cittadella', 'Clermont Foot 63', 'Club América', 'Club Atlas', 'Club Atlético Aldosivi', 'Club Atlético Banfield', 'Club Atlético Colón', 'Club Atlético Huracán', 'Club Atlético Lanús', 'Club Atlético Talleres', 'Club Atlético Tigre', 'Club Brugge KV', 'Club Deportes Temuco', 'Club León', 'Club Necaxa', 'Club Tijuana', 'Clube Sport Marítimo', 'Colchester United', 'Colo-Colo', 'Colorado Rapids', 'Columbus Crew SC', 'Cork City', 'Cosenza', 'Coventry City', 'Cracovia', 'Crawley Town', 'Crewe Alexandra', 'Crotone', 'Cruz Azul', 'Crystal Palace', 'Curicó Unido', 'Cádiz CF', 'Córdoba CF', 'DC United', 'DSC Arminia Bielefeld', 'Daegu FC', 'Dalian YiFang FC', 'Dalkurd FF', 'De Graafschap', 'Defensa y Justicia', 'Deportes Iquique', 'Deportes Tolima', 'Deportivo Alavés', 'Deportivo Cali', 'Deportivo Pasto', 'Deportivo Toluca', 'Deportivo de La Coruña', 'Derby County', 'Dijon FCO', 'Dinamo Zagreb', 'Djurgårdens IF', 'Doncaster Rovers', 'Dundalk', 'Dundee FC', 'Dynamo Kyiv', 'ESTAC Troyes', 'Eintracht Braunschweig', 'Eintracht Frankfurt', 'Empoli', 'En Avant de Guingamp', 'Envigado FC', 'Esbjerg fB', 'Ettifaq FC', 'Everton', 'Excelsior', 'Exeter City', 'FC Admira Wacker Mödling', 'FC Augsburg', 'FC Barcelona', 'FC Basel 1893', 'FC Bayern München', 'FC Carl Zeiss Jena', 'FC Dallas', 'FC Emmen', 'FC Energie Cottbus', 'FC Erzgebirge Aue', 'FC Girondins de Bordeaux', 'FC Groningen', 'FC Hansa Rostock', 'FC Ingolstadt 04', 'FC København', 'FC Lorient', 'FC Lugano', 'FC Luzern', 'FC Metz', 'FC Midtjylland', 'FC Nantes', 'FC Nordsjælland', 'FC Porto', 'FC Red Bull Salzburg', 'FC Schalke 04', 'FC Seoul', 'FC Sion', 'FC Sochaux-Montbéliard', 'FC St. Gallen', 'FC St. Pauli', 'FC Thun', 'FC Tokyo', 'FC Utrecht', 'FC Wacker Innsbruck', 'FC Würzburger Kickers', 'FC Zürich', 'FK Austria Wien', 'FK Bodø/Glimt', 'FK Haugesund', 'FSV Zwickau', 'Fenerbahçe SK', 'Feyenoord', 'Fiorentina', 'Fleetwood Town', 'Foggia', 'Forest Green Rovers', 'Fortuna Düsseldorf', 'Fortuna Sittard', 'Frosinone', 'Fulham', 'GD Chaves', 'GFC Ajaccio', 'GIF Sundsvall', 'Galatasaray SK', 'Gamba Osaka', 'Gangwon FC', 'Genoa', 'Getafe CF', 'Gillingham', 'Gimnasia y Esgrima La Plata', 'Gimnàstic de Tarragona', 'Girona FC', 'Godoy Cruz', 'Granada CF', 'Grasshopper Club Zürich', 'Grenoble Foot 38', 'Grimsby Town', 'Guadalajara', 'Guangzhou Evergrande Taobao FC', 'Guangzhou R&F; FC', 'Guizhou Hengfeng FC', 'Gyeongnam FC', 'Górnik Zabrze', 'Göztepe SK', 'HJK Helsinki', 'Hallescher FC', 'Hamburger SV', 'Hamilton Academical FC', 'Hammarby IF', 'Hannover 96', 'Heart of Midlothian', 'Hebei China Fortune FC', 'Hellas Verona', 'Henan Jianye FC', 'Heracles Almelo', 'Hertha BSC', 'Hibernian', 'Hobro IK', 'Hokkaido Consadole Sapporo', 'Holstein Kiel', 'Houston Dynamo', 'Huddersfield Town', 'Hull City', 'IF Brommapojkarna', 'IF Elfsborg', 'IFK Göteborg', 'IFK Norrköping', 'IK Sirius', 'IK Start', 'Incheon United FC', 'Independiente', 'Independiente Medellín', 'Independiente Santa Fe', 'Inter', 'Ipswich Town', 'Itagüí Leones FC', 'Jagiellonia Białystok', 'Jaguares de Córdoba', 'Jeju United FC', 'Jeonbuk Hyundai Motors', 'Jeonnam Dragons', 'Jiangsu Suning FC', 'Junior FC', 'Juventus', 'Júbilo Iwata', 'KAA Gent', 'KAS Eupen', 'KFC Uerdingen 05', 'KRC Genk', 'KSV Cercle Brugge', 'KV Kortrijk', 'KV Oostende', 'Kaizer Chiefs', 'Kalmar FF', 'Karlsruher SC', 'Kashima Antlers', 'Kashiwa Reysol', 'Kasimpaşa SK', 'Kawasaki Frontale', 'Kayserispor', 'Kilmarnock', 'Korona Kielce', 'Kristiansund BK', 'LA Galaxy', 'LASK Linz', 'LOSC Lille', 'La Berrichonne de Châteauroux', 'La Equidad', 'Lazio', 'Le Havre AC', 'Lecce', 'Lech Poznań', 'Lechia Gdańsk', 'Leeds United', 'Legia Warszawa', 'Leicester City', 'Levante UD', 'Lillestrøm SK', 'Limerick FC', 'Lincoln City', 'Liverpool', 'Livingston FC', 'Livorno', 'Lobos BUAP', 'Lokomotiv Moscow', 'Los Angeles FC', 'Luton Town', 'MKE Ankaragücü', 'MSV Duisburg', 'Macclesfield Town', 'Malmö FF', 'Manchester City', 'Manchester United', 'Mansfield Town', 'Medipol Başakşehir FK', 'Melbourne City FC', 'Melbourne Victory', 'Middlesbrough', 'Miedź Legnica', 'Milan', 'Millonarios FC', 'Millwall', 'Milton Keynes Dons', 'Minnesota United FC', 'Molde FK', 'Monarcas Morelia', 'Monterrey', 'Montpellier HSC', 'Montreal Impact', 'Morecambe', 'Moreirense FC', 'Motherwell', 'Málaga CF', 'NAC Breda', 'Nagoya Grampus', 'Napoli', 'Neuchâtel Xamax', 'New England Revolution', 'New York City FC', 'New York Red Bulls', 'Newcastle Jets', 'Newcastle United', "Newell's Old Boys", 'Newport County', 'Northampton Town', 'Norwich City', 'Nottingham Forest', 'Notts County', 'Nîmes Olympique', 'OGC Nice', 'Odds BK', 'Odense Boldklub', 'Ohod Club', 'Oldham Athletic', 'Olympiacos CFP', 'Olympique Lyonnais', 'Olympique de Marseille', 'Once Caldas', 'Orlando City SC', 'Orlando Pirates', 'Os Belenenses', 'Oxford United', 'PAOK', 'PEC Zwolle', 'PFC CSKA Moscow', 'PSV', 'Pachuca', 'Padova', 'Palermo', 'Panathinaikos FC', 'Paris FC', 'Paris Saint-Germain', 'Parma', 'Patriotas Boyacá FC', 'Patronato', 'Perth Glory', 'Perugia', 'Pescara', 'Peterborough United', 'Philadelphia Union', 'Piast Gliwice', 'Plymouth Argyle', 'Pogoń Szczecin', 'Pohang Steelers', 'Port Vale', 'Portimonense SC', 'Portland Timbers', 'Portsmouth', 'Preston North End', 'Puebla FC', 'Queens Park Rangers', 'Querétaro', 'RB Leipzig', 'RC Celta', 'RC Strasbourg Alsace', 'RCD Espanyol', 'RCD Mallorca', 'RSC Anderlecht', 'Racing Club', 'Racing Club de Lens', 'Randers FC', 'Rangers FC', 'Ranheim Fotball', 'Rayo Vallecano', 'Reading', 'Real Betis', 'Real Madrid', 'Real Oviedo', 'Real Salt Lake', 'Real Sociedad', 'Real Sporting de Gijón', 'Real Valladolid CF', 'Real Zaragoza', 'Red Star FC', 'Rio Ave FC', 'Rionegro Águilas', 'River Plate', 'Rochdale', 'Roma', 'Rosario Central', 'Rosenborg BK', 'Rotherham United', 'Royal Antwerp FC', 'Royal Excel Mouscron', 'SC Braga', 'SC Fortuna Köln', 'SC Freiburg', 'SC Heerenveen', 'SC Paderborn 07', 'SC Preußen Münster', 'SCR Altach', 'SD Eibar', 'SD Huesca', 'SG Dynamo Dresden', 'SG Sonnenhof Großaspach', 'SK Brann', 'SK Rapid Wien', 'SK Slavia Praha', 'SK Sturm Graz', 'SKN St. Pölten', 'SL Benfica', 'SPAL', 'SV Darmstadt 98', 'SV Mattersburg', 'SV Meppen', 'SV Sandhausen', 'SV Wehen Wiesbaden', 'SV Werder Bremen', 'SV Zulte-Waregem', 'Sagan Tosu', 'Sampdoria', 'San Jose Earthquakes', 'San Lorenzo de Almagro', 'San Luis de Quillota', 'San Martin de Tucumán', 'San Martín de San Juan', 'Sandefjord Fotball', 'Sanfrecce Hiroshima', 'Sangju Sangmu FC', 'Santos Laguna', 'Sarpsborg 08 FF', 'Sassuolo', 'Scunthorpe United', 'Seattle Sounders FC', 'Sevilla FC', 'Shakhtar Donetsk', 'Shamrock Rovers', 'Shandong Luneng TaiShan FC', 'Shanghai Greenland Shenhua FC', 'Shanghai SIPG FC', 'Sheffield United', 'Sheffield Wednesday', 'Shimizu S-Pulse', 'Shonan Bellmare', 'Shrewsbury', 'Sint-Truidense VV', 'Sivasspor', 'Sligo Rovers', 'Southampton', 'Southend United', 'SpVgg Greuther Fürth', 'SpVgg Unterhaching', 'Sparta Praha', 'Spartak Moscow', 'Spezia', 'Sporting CP', 'Sporting Kansas City', 'Sporting Lokeren', 'Sporting de Charleroi', 'St. Johnstone FC', 'St. Mirren', "St. Patrick's Athletic", 'Stabæk Fotball', 'Stade Brestois 29', 'Stade Malherbe Caen', 'Stade Rennais FC', 'Stade de Reims', 'Standard de Liège', 'Stevenage', 'Stoke City', 'Strømsgodset IF', 'Sunderland', 'Suwon Samsung Bluewings', 'Swansea City', 'Swindon Town', 'Sydney FC', 'SønderjyskE', 'TSG 1899 Hoffenheim', 'TSV 1860 München', 'TSV Hartberg', 'Tianjin Quanjian FC', 'Tianjin TEDA FC', 'Tiburones Rojos de Veracruz', 'Tigres U.A.N.L.', 'Torino', 'Toronto FC', 'Tottenham Hotspur', 'Toulouse Football Club', 'Trabzonspor', 'Tranmere Rovers', 'Trelleborgs FF', 'Tromsø IL', 'U.N.A.M.', 'UD Almería', 'UD Las Palmas', 'US Cremonese', 'US Orléans Loiret Football', 'US Salernitana 1919', 'Udinese', 'Ulsan Hyundai FC', 'Universidad Católica', 'Universidad de Chile', 'Unión Española', 'Unión La Calera', 'Unión de Santa Fe', 'Urawa Red Diamonds', 'V-Varen Nagasaki', 'VVV-Venlo', 'Valencia CF', 'Valenciennes FC', 'Vancouver Whitecaps FC', 'Vegalta Sendai', 'Vejle Boldklub', 'Vendsyssel FF', 'Venezia FC', 'VfB Stuttgart', 'VfL Bochum 1848', 'VfL Osnabrück', 'VfL Sportfreunde Lotte', 'VfL Wolfsburg', 'VfR Aalen', 'Viktoria Plzeň', 'Villarreal CF', 'Vissel Kobe', 'Vitesse', 'Vitória Guimarães', 'Vitória de Setúbal', 'Vålerenga Fotball', 'Vélez Sarsfield', 'Waasland-Beveren', 'Walsall', 'Waterford FC', 'Watford', 'Wellington Phoenix', 'West Bromwich Albion', 'West Ham United', 'Western Sydney Wanderers', 'Wigan Athletic', 'Willem II', 'Wisła Kraków', 'Wisła Płock', 'Wolfsberger AC', 'Wolverhampton Wanderers', 'Wycombe Wanderers', 'Yeni Malatyaspor', 'Yeovil Town', 'Yokohama F. Marinos', 'Zagłębie Lubin', 'Zagłębie Sosnowiec', 'Çaykur Rizespor', 'Örebro SK', 'Östersunds FK', 'Śląsk Wrocław'])



#if st.button('Mostrar filtro seleccionado'):
#    st.write("Filtro seleccionado:", sub_option)

if option == 'Por Nacionalidad':
    filtered_data = data[data['Nationality'] == sub_option]
elif option == 'Por Posicion':
    filtered_data = data[data['Position'] == sub_option]
else:
    filtered_data = data[data['Club'] == sub_option]



#filtered = data[data['Nationality'] == 'France']

#names = list(filtered["Name"])
ids = list(filtered_data["realidnumber"])
#st.write(ids)

#if st.button('Mostrar lista:'):
#    st.write("Lista:", ids)



import streamlit as st
import random
import numpy as np

# Assuming model_interp, vae_loaded, and affect are defined in your environment
def show_interp3(ids, scale=1.5):
    """Display a list of images in Streamlit for 5 random interpolations."""
    for _ in range(5):  # Generate 5 sets of random interpolations
        # Select two random indices
        index1 = random.choice(ids)
        index2 = random.choice([num for num in ids if num != index1])
        
        # Generate interpolated images
        interp_result = model_interp(model=vae_loaded, index1=index1, index2=index2).unbind(0)
        
        # Convert tensors to numpy arrays and prepare for display
        imgs = [img.permute(1, 2, 0).cpu() for img in interp_result]  # Assuming this converts to (H, W, C)
        nrg = 8

        # Create columns layout
        num_columns = len(imgs)  # One column per image
        columns = st.columns(num_columns)  # Create a column for each image
        
        for i, (col, img) in enumerate(zip(columns, imgs)):
            try:
                img = img.detach().numpy()
            except:
                pass

            # Ensure image format (H, W, C)
            if isinstance(img, np.ndarray):
                img = img.transpose(0, 1, 2)  # Convert (C, H, W) -> (H, W, C)

            # Handle cases with more than 3 channels
            if img.shape[2] > 3:
                img = img[:, :, :3]  # Only keep RGB channels

            # Display each image with scaling in the respective column
            with col:
                if i == 0:
                    col.image(
                        affect[index1][0].permute(1, 2, 0).cpu().detach().numpy(),
                        caption=f"{affect[index1][1]}",
                        width=int(150 * scale)
                    )
                    #st.empty()
                elif i == len(imgs) - 1:
                    col.image(
                        affect[index2][0].permute(1, 2, 0).cpu().detach().numpy(),
                        caption=f"{affect[index2][1]}",
                        width=int(150 * scale)
                    )
                    #st.empty()
                elif i == nrg:
                    col.image(img, caption="Jugador Generado", width=int(150 * scale))


if st.button("Generar jugadores"):
    show_interp3(ids)

