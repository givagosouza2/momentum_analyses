import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import signal
import scipy.interpolate
from scipy.signal import find_peaks
import re
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import tempfile
from scipy.fft import fft
from scipy.spatial.distance import pdist, squareform
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.lib.utils import ImageReader
import datetime


def individual_info_balance():
    name = st.text_input("Avaliação do equilíbrio estático - Nome:")
    minDate = datetime.datetime(1900, 1, 1)
    date = st.date_input(
        "Avaliação do equilíbrio estático - Data de teste", value=None, min_value=minDate)
    doctor = st.text_input(
        "Avaliação do equilíbrio estático - Encaminhamento:")
    birthdate = st.date_input(
        "Avaliação do equilíbrio estático - Data de nascimento", value=None, min_value=minDate)
    options = ["Feminino", "Masculino"]
    sex = st.selectbox("Avaliação do equilíbrio estático - Sexo:", options)
    contact = st.text_input("Avaliação do equilíbrio estático - Contato:")
    return name, date, doctor, birthdate, sex, contact


def individual_info_iTUG():
    name = st.text_input("Avaliação da mobilidade - Nome:")
    minDate = datetime.datetime(1900, 1, 1)
    date = st.date_input(
        "Avaliação da mobilidade - Data de teste", value=None, min_value=minDate)
    doctor = st.text_input("Avaliação da mobilidade - Encaminhamento:")
    birthdate = st.date_input(
        "Avaliação da mobilidade - Data de nascimento", value=None, min_value=minDate)
    options = ["Feminino", "Masculino"]
    sex = st.selectbox("Avaliação da mobilidade - Sexo:", options)
    contact = st.text_input("Avaliação da mobilidade - Contato:")
    return name, date, doctor, birthdate, sex, contact


def individual_info_FTT():
    name = st.text_input("Avaliação da coordenação motora - Nome:")
    minDate = datetime.datetime(1900, 1, 1)
    date = st.date_input(
        "Avaliação da coordenação motora - Data de teste", value=None, min_value=minDate)
    doctor = st.text_input("Avaliação da coordenação motora - Encaminhamento:")
    birthdate = st.date_input(
        "Avaliação da coordenação motora - Data de nascimento", value=None, min_value=minDate)
    options = ["Feminino", "Masculino"]
    sex = st.selectbox("Avaliação da coordenação motora - Sexo:", options)
    contact = st.text_input("Avaliação da coordenação motora - Contato:")
    return name, date, doctor, birthdate, sex, contact


def individual_info_tremor():
    name = st.text_input("Avaliação do tremor - Nome:")
    minDate = datetime.datetime(1900, 1, 1)
    date = st.date_input(
        "Avaliação do tremor - Data de teste", value=None, min_value=minDate)
    doctor = st.text_input("Avaliação do tremor - Encaminhamento:")
    birthdate = st.date_input(
        "Avaliação do tremor - Data de nascimento", value=None, min_value=minDate)
    options = ["Feminino", "Masculino"]
    sex = st.selectbox("Avaliação do tremor - Sexo:", options)
    contact = st.text_input("Avaliação do tremor - Contato:")
    return name, date, doctor, birthdate, sex, contact


def rms_amplitude(time_series):
    chunk = time_series
    # Square each value
    squared_values = [x**2 for x in chunk]
    # Calculate the mean of the squared values
    mean_squared = np.mean(squared_values)
    # Take the square root of the mean
    return mean_squared


def approximate_entropy2(data, n, r):
    correlation = np.zeros(2)
    for m in range(n, n+2):  # Run it twice, with window size differing by 1
        set = 0
        count = 0
        counter = np.zeros(len(data) - m + 1)
        window_correlation = np.zeros(len(data) - m + 1)

        for i in range(0, len(data) - m + 1):
            # Current window stores the sequence to be compared with other sequences
            current_window = data[i:i + m]

            for j in range(0, len(data) - m + 1):
                # Get a window for comparison with the current_window
                sliding_window = data[j:j + m]

                for k in range(m):
                    if (abs(current_window[k] - sliding_window[k]) > r) and set == 0:
                        set = 1  # The difference between the two sequences is greater than the given value

                if set == 0:
                    count += 1  # Measure how many sliding_windows are similar to the current_window

                    set = 0  # Reset 'set'

            # Number of similar windows for every current_window
            counter[i] = count / (len(data) - m + 1)
            count = 0

        correlation[m - n] = np.sum(counter) / (len(data) - m + 1)

    apen = np.log(correlation[0] / correlation[1])
    return apen


def ellipse_model(x, y):
    centro_u = np.mean(x)
    centro_v = np.mean(y)

    for j in range(1):
        P = np.column_stack((x, y))
        K = ConvexHull(P).vertices
        K = np.unique(K)
        PK = P[K].T
        d, N = PK.shape
        Q = np.zeros((d + 1, N))
        Q[:d, :] = PK[:d, :N]
        Q[d, :] = np.ones(N)

        count = 1
        err = 1
        u = (1 / N) * np.ones(N)
        tolerance = 0.01

        while err > tolerance:
            X = Q @ np.diag(u) @ Q.T
            M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
            new_u = (1 - step_size) * u
            new_u[j] = new_u[j] + step_size
            count = count + 1
            err = np.linalg.norm(new_u - u)
            u = new_u

        U = np.diag(u)
        A = (1 / d) * np.linalg.inv(PK @ U @ PK.T - (PK @ u) @ (PK @ u).T)
        c = PK @ u
        U, Q, V = np.linalg.svd(A)
        r1 = 1 / np.sqrt(Q[0])
        r2 = 1 / np.sqrt(Q[1])
        v = np.array([r1, r2, c[0], c[1], V[0, 0]])

        D = v[1]
        d = v[0]
        tan = centro_v / centro_u
        arco = np.arctan(tan)
        rot = arco
        angle = np.arange(0, 2 * np.pi, 2 * np.pi / 360)
        u = centro_u + (D * np.cos(angle) * np.cos(rot)) - \
            (d * np.sin(angle) * np.sin(rot))
        v = centro_v + (D * np.cos(angle) * np.sin(rot)) + \
            (d * np.sin(angle) * np.cos(rot))
        phi = rot * 180 / np.pi
        DM = D
        dm = d
        q = len(x)
        positive = 0
        number_of_true = len(angle)

        while number_of_true/q > 0.95:
            u = centro_u + (DM * np.cos(angle) * np.cos(rot)) - \
                (dm * np.sin(angle) * np.sin(rot))
            v = centro_v + (DM * np.cos(angle) * np.sin(rot)) + \
                (dm * np.sin(angle) * np.cos(rot))
            vertices = zip(u, v)
            polygon = Polygon(vertices)
            for xi, yi in zip(x, y):
                points = Point(xi, yi)
                if polygon.contains(points) == True:
                    positive = positive + 1
            number_of_true = positive
            positive = 0
            DM = DM - DM*0.2
            dm = dm - dm*0.2
    return u, v, DM, dm, phi

# fast fourier transform for spectral analysis


def tremor_fft(data):
    fs = 100  # Sampling frequency
    fft_results = []
    frequencies = []

    # Perform FFT
    fft_result = np.abs(fft(data))
    fft_result[0] = 0
    N = len(fft_result)
    freq = np.fft.fftfreq(N, 1/fs)

    pos = 0
    for i in freq:
        if i >= 0:
            pos = pos + 1
        else:
            f = pos
            break
    power_spectrum = fft_result[0:f]
    temp_freq = freq[0:f]

    return power_spectrum, temp_freq


def balance_fft(data):
    fs = 100  # Sampling frequency
    # Perform FFT
    fft_result = np.fft.fft(data)
    N = len(fft_result)

    frequencies = np.fft.fftfreq(N, 1/fs)
    spectrum_amplitude = []
    spectrum_amplitude = (np.abs(fft_result/N))
    spectrum_amplitude[0] = 0
    freq = []
    spectra = []
    a = 0
    c = 0
    for i in frequencies:
        if i >= 0:
            freq.append(i)
            spectra.append(spectrum_amplitude[c])
            a = a + 1
        c = c + 1

    a = 0
    for i in freq:
        a = a + 1
        if i > 0.5:
            f1 = a
            break
    a = 0
    for i in freq:
        a = a + 1
        if i > 2:
            f2 = a
            break
    a = 0
    for i in freq:
        a = a + 1
        if i > 6:
            f3 = a
            break

    total_spectral_energy = sum(spectra[0:f3])
    energy = 0

    c = 1
    while energy < total_spectral_energy/2:
        energy = np.sum(spectra[0:c])
        c = c + 1
    median_frequency = freq[c]
    LF_energy = sum(spectra[0:f1])
    MF_energy = sum(spectra[f1:f2])
    HF_energy = sum(spectra[f2:f3])

    return freq, spectra, median_frequency, LF_energy, MF_energy, HF_energy

# function for butterworth filter


def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype, analog=False)
    y = filtfilt(b, a, data)
    return y

# function to fit the ellipse in the statokinesiogram plot


def set_ellipse(fpML, fpAP):
    points = np.column_stack((fpML, fpAP))
    hull = ConvexHull(points)

    # Get the boundary points of the convex hull
    boundary_points = points[hull.vertices]

    # Calculate the centroid of the boundary points
    centroidx = np.mean(fpML)
    centroidy = np.mean(fpAP)
    centroid = centroidx, centroidy

    # Calculate the covariance matrix of the boundary points
    covariance = np.cov(boundary_points, rowvar=False)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Calculate the major and minor axis of the ellipse
    major_axis = np.sqrt(eigenvalues[0]) * np.sqrt(-2 * np.log(1 - 0.95))/2
    minor_axis = np.sqrt(eigenvalues[1]) * np.sqrt(-2 * np.log(1 - 0.95))/2

    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    area = np.pi*major_axis*minor_axis
    num_points = 101  # 360/100 + 1
    ellipse_points = np.zeros((num_points, 2))
    a = 0
    for i in np.arange(0, 361, 360 / 100):
        ellipse_points[a, 0] = centroid[0] + major_axis * np.cos(np.radians(i))
        ellipse_points[a, 1] = centroid[1] + minor_axis * np.sin(np.radians(i))
        a += 1
    angle_deg = -angle
    angle_rad = np.radians(angle_deg)

    # Matrix for ellipse rotation
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    ellipse_points = np.dot(ellipse_points, R)
    return ellipse_points, area, angle_deg, major_axis, minor_axis


# Set the page expanded configuration with two tabs
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
limite = number = st.number_input("Insert a number")
init, tab1, tab2, tab3, tab4, infoUs, videos, contact = st.tabs(
    ["Projeto Momentum", "Equilíbrio estático", "iTUG", "Finger tapping test", "Tremor de mão", "Quem somos nós?", "Vídeos", "Contate-nos"])
with init:

    # Agora você pode adicionar o conteúdo do seu site
    st.write("<h1 style='text-align: center; color: blue'; >Projeto Momentum</h1>",
             unsafe_allow_html=True)
    t1, t2 = st.columns([1, 1])
    with t1:
        st.markdown(
            '<h2 style="text-align: center; color: blue;">O que é o projeto Momentum?</h2>', unsafe_allow_html=True)
        paragraph = """
        <p style="text-align: justify;">
            O projeto Momentum é uma iniciativa acadêmico-científica que visa unir a formação de recursos humanos, criação do conhecimento e desenvolvimento tecnológico em prol de oferecer uma ferramenta de baixo custo, mas confiável que pudesse ser usada para avaliar a saúde das pessoas.
            Para isso, pesquisadores de diferentes instituições de ensino superior públicas brasileira pensaram em desenvolver propostas de protocolos de avaliação em saúde que pudessem ser feitas usando sensores eletrônicos presentes no interior de smartphones.
        </p>        
        """
        st.markdown(paragraph, unsafe_allow_html=True)

        st.markdown(
            '<h2 style="text-align: center; color: blue;">Histórico</h2>', unsafe_allow_html=True)
        paragraph = """
        <p style="text-align: justify;">            
            Os primeiros passos desse projeto foram dados a partir da convergência de ideias de docentes da Universidade Federal do Pará (Anselmo de Athayde Costa e Silva, Bianca Callegari, Givago da Silva Souza, Gustavo Henrique Lima), da Universidade do Estado do Pará (André dos Santos Cabral) e do Instituto Federal de São Paulo (Anderson Belgamo).             
        </p>
        <p style="text-align: justify;">            
            Os primeiros protótipos de aplicativos do projeto Momentum foram desenvolvidos pelo professor Anderson Belgamo e serviu de base para o primeiro aplicativo do projeto Momentum, o qual foi desenvolvido por um discente da Faculdade de Ciências da Computação da Universidade Federal do Pará, Enzo Gabriel Rocha dos Santos, dentro do programa de iniciação científica da Universidade Federal do Pará. Esse aplicativo foi denomindado de Momentum Science e foi registrado no Instituto Nacional de Propriedade Industrial sob o número #XXXXXXXXXX.  
        </p> 
        <p style="text-align: justify;">            
            O segundo aplicativo do projeto Momentum foi desenvolvido pelo discente Felipe André da Costa Brito do curso de Doutorado do Programa de Neurociências e Biologia Celular da Universidade Federal do Pará. Esse aplicativo foi denominado de Momentum Touch e o seu registro está em análise no Instituto Nacional de Propriedade Intelectual.  
        </p>    
        """
        st.markdown(paragraph, unsafe_allow_html=True)

        image = st.image("image2.jpeg")

        st.markdown(
            '<h2 style="text-align: center;color: blue;">Análise de dados dos aplicativos do projeto Momentum</h2>', unsafe_allow_html=True)
        paragraph = """
        <p style="text-align: justify;">            
            Esta página traz uma série de rotinas que podem ser úteis para os usuários dos aplicativos do projeto Momentum. Essas rotinas foram escritas com o intuito de auxiliar na análise de dados dos projetos de pesquisa dos desenvolvedores dos aplicativos e que podem ser usadas de forma gratuita por usuários dos aplicativos.
        </p>
        <p style="text-align: justify;">            
        Todas as rotinas foram escritas em linguagem Python e só precisam que os arquivos de saída dos aplicativos sejam carregados para a rotina e uma análise sobre os dados será realizada, sendo mostrado representação gráfica e quantitativas das características de importância das diferentes tarefas estudadas. As rotinas presentes aqui são para a análise do equilíbrio estático, da mobilidade (instrumented Timed Up and Go - iTUG), da coordenação motora (Finger Tapping Test - FTT) e tremor de mão.
        </p>
        """
        st.markdown(paragraph, unsafe_allow_html=True)

    with t2:
        image = st.image("image1.jpeg")
        st.markdown(
            '<h2 style="text-align: center; color: blue;">O que fazem esses aplicativos?</h2>', unsafe_allow_html=True)
        paragraph = """
        <p style="text-align: justify;">            
            O Momentum Science é um aplicativo que salva as leituras dos sensores inerciais presentes no smartphone. Os sinais inerciais capturados usando o smartphone podem ser úteis para caracterizar o movimento humano em diferentes tarefas. O Momentum Touch é um aplicativo que captura a leitura da tela sensível ao toque do smartphone durante um teste de coordenação motora chamado Finger Tapping test.
        </p>
        """
        st.markdown(paragraph, unsafe_allow_html=True)

        st.markdown(
            '<h2 style="text-align: center;color: blue;">Publicações científicas usando aplicativos do Projeto Momentum</h2>', unsafe_allow_html=True)
        paragraph = """
        <p style="text-align: justify;">            
            1. RODRIGUES, L. A. ; SANTOS, E. G. R. ; SANTOS, P. S. A. ; IGARASHI, Y. ; OLIVEIRA, L. K. R. ; PINTO, G. H. L. ; SANTOS-LOBATO, B. L. ; CABRAL, A. S. ; BELGAMO, A. ; COSTA E SILVA, A. A ; CALLEGARI, B. ; Souza, G. S. . Wearable Devices and Smartphone Inertial Sensors for Static Balance Assessment: A Concurrent Validity Study in Young Adult Population. Journal Of Personalized Medicine, v. 1, p. 1-1, 2022.            
        </p>        
        """
        st.markdown(paragraph, unsafe_allow_html=True)
        st.write("[Link para o artigo](https://www.mdpi.com/2075-4426/12/7/1019)")
        paragraph = """
        <p style="text-align: justify;">            
            2. DUARTE, M. B. ; MORAES, A. A. C. ; FERREIRA, E. V. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; OLIVEIRA, P. R. ; AMORIM, C. F. ; ANDRÉ DOS SANTOS, CABRAL ; SAUNIER, G. J. A. ; COSTA E SILVA, A. A. ; SOUZA, GIVAGO S. ; CALLEGARI, B. . Validity and reliability of a smartphone-based assessment for anticipatory and compensatory postural adjustments during predictable perturbations. GAIT & POSTURE, v. 96, p. 9-17, 2022.
        </p>
        """
        st.markdown(paragraph, unsafe_allow_html=True)
        st.write(
            "[Link para o artigo](https://linkinghub.elsevier.com/retrieve/pii/S0966636222001278)")
        paragraph = """
        <p style="text-align: justify;">            
            3. MORAES, A. A. C. ; DUARTE, M. B. ; FERREIRA, E. V. ; ALMEIDA, G. C. S. ; SANTOS, E. G. R. ; PINTO, G. H. L. ; OLIVEIRA, P. R. ; AMORIM, C. F. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A. ; Souza, G. S. ; CALLEGARI, B. . Validity and reliability of smartphone app for evaluating postural adjustments during step initiation. SENSORS, v. 1, p. 1, 2022.
        </p>
        """
        st.markdown(paragraph, unsafe_allow_html=True)
        st.write("[Link para o artigo](https://www.mdpi.com/1424-8220/22/8/2935)")
        paragraph = """
        <p style="text-align: justify;">            
            4. BRITO, F. A. C. ; MONTEIRO, L. C. P. ; SANTOS, E. G. R. ; LIMA, R. C. ; SANTOS-LOBATO, B. L. ; ANDRÉ DOS SANTOS, CABRAL ; CALLEGARI, B. ; SILVA, A. A. C. E. ; GIVAGO DA SILVA, SOUZA . The role of sex and handedness in the performance of the smartphone-based Finger-Tapping Test. PLOS Digital Health, v. 2, p. e0000304, 2023.
        </p>
        """
        st.markdown(paragraph, unsafe_allow_html=True)
        st.write(
            "[Link para o artigo](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000304)")
        paragraph = """
        <p style="text-align: justify;">            
            5. MORAES, A. A. C. ; DUARTE, M. B. ; SANTOS, E. J. M. ; ALMEIDA, G. C. S. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A. ; GARCEZ, D. R. ; GIVAGO DA SILVA, SOUZA ; CALLEGARI, B. . Comparison of inertial records during anticipatory postural adjustments obtained with devices of different masses. PeerJ, v. 11, p. e15627, 2023.
        </p>
        """
        st.markdown(paragraph, unsafe_allow_html=True)
        st.write("[Link para o artigo](https://peerj.com/articles/15627/)")
        paragraph = """
        <p style="text-align: justify;">            
            6. SANTOS, P. S. A. ; SANTOS, E. G. R. ; MONTEIRO, L. C. P. ; SANTOS-LOBATO, B. L. ; PINTO, G. H. L. ; BELGAMO, A. ; ANDRÉ DOS SANTOS, CABRAL ; COSTA E SILVA, A. A ; CALLEGARI, B. ; SOUZA, Givago da Silva . The hand tremor spectrum is modified by the inertial sensor mass during lightweight wearable and smartphone-based assessment in healthy young subjects. Scientific Reports, v. 12, p. 01, 2022.
        </p>
        """
        st.markdown(paragraph, unsafe_allow_html=True)
        st.write(
            "[Link para o artigo](https://www.nature.com/articles/s41598-022-21310-4)")

    footer = """
<div style="text-align: center;">
    <p>Projeto Momentum. Todos os direitos reservados.</p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

# Algorithm to proceed static balance control analysis
with tab1:
    st.markdown(
        '<h2 style="text-align: center; color: blue;">Avaliação do equilíbrio estático em smartphone</h2>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns([1, 1, 1])

    # Acceleration file upload button
    uploaded_acc = st.file_uploader(
        "Selecione o aquivo de texto do acelerômetro", type=["txt"],)

    # Check if a file has been uploaded
    if uploaded_acc is not None:

        # Read and display the data from the uploaded CSV file
        if uploaded_acc is not None:
            st.markdown(
                '<h4 style="text-align: Left; color: blue;">Informação da pessoa testada</h4>', unsafe_allow_html=True)
            name_participant, date_participant, doctor_participant, birthdate_participant, sex_participant, contact_participant = individual_info_balance()
            st.markdown(
                '<h4 style="text-align: Left; color: blue;">Informação do teste realizado</h4>', unsafe_allow_html=True)
            options = ["Olhos abertos", "Olhos fechados"]
            test_condition = st.selectbox("Visualização:", options)
            options = ["Bipedal", "Unipedal"]
            support = st.selectbox("Suporte:", options)
            duration = st.text_input("Duração do teste (s)")

            custom_separator = ';'

            # Allocation of the data to the variables
            df = pd.read_csv(uploaded_acc, sep=custom_separator)
            t = df.iloc[:, 0]
            x = df.iloc[:, 1]
            y = df.iloc[:, 2]
            z = df.iloc[:, 3]
            time = t
            AP = z

            # Selection of ML axis between x and y axis. The one that recorded the gravity acceleration is excluded and the other is the ML axis
            if np.mean(x) > np.mean(y):
                ML = y
            else:
                ML = x

                # Pre-processing data: AP and ML channels were detrended and normalized to gravity acceleration
            if np.max(x) > 9 or np.max(y) > 9:
                AP = signal.detrend(AP/9.81)
                ML = signal.detrend(ML/9.81)
            else:
                AP = signal.detrend(AP)
                ML = signal.detrend(ML)

                # Pre-processing data: interpolating to 100 Hz
            interpf = scipy.interpolate.interp1d(time, AP)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            AP_ = interpf(time_)
            xAP, yAP = time_/1000, AP_
            yAP = butterworth_filter(yAP, 6, 100, order=4, btype='low')
            interpf = scipy.interpolate.interp1d(time, ML)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            ML_ = interpf(time_)
            xML, yML = time_/1000, ML_
            yML = butterworth_filter(yML, 6, 100, order=4, btype='low')

            # norm calculation
            norm = np.sqrt(yAP**2+yML**2)
            length_balance = len(yML)

            # Creating controls to interacts with the plots
            with t3:
                slider_min = st.number_input(
                    "Selecione o início do registro", min_value=1, max_value=length_balance-1, step=1, value=1)
                slider_max = st.number_input(
                    "Selecione o final do registro", min_value=1, max_value=length_balance-1, value=length_balance-1, step=1)
                initial_state = True
                checkbox_1 = st.checkbox(
                    "Mostrar o registro completo", value=initial_state)
                checkbox_2 = st.checkbox(
                    "Mostrar o período de análise do registro", value=initial_state)

                # Ellipse fitting and features extraction from ellipse
            ellipse_fit, area_value, angle_deg_value, major_axis_value, minor_axis_value = set_ellipse(
                yML[slider_min:slider_max], yAP[slider_min:slider_max])

            # Extracting features: total deviation, rmsAP, rmsML
            total_deviation = sum(np.sqrt(yAP[slider_min:slider_max]**2+yML[slider_min:slider_max]**2))
            rmsAP = np.sqrt(np.mean(np.square(yAP[slider_min:slider_max])))
            rmsML = np.sqrt(np.mean(np.square(yML[slider_min:slider_max])))

            frequencies, spectrum_amplitude_ML, median_frequency_ML, LF_energy_ML, MF_energy_ML, HF_energy_ML = balance_fft(
                yML[slider_min:slider_max])
            frequencies, spectrum_amplitude_AP, median_frequency_AP, LF_energy_AP, MF_energy_AP, HF_energy_AP = balance_fft(
                yAP[slider_min:slider_max])

            maxX = np.max(ellipse_fit[:, 0])
            maxY = np.max(ellipse_fit[:, 1])
            maxValue = np.max([maxX, maxY])
            if maxValue <= 0.1:
                lim = 0.1
            elif maxValue > 0.1 and maxValue < 0.3:
                lim = 0.3
            elif maxValue > 0.3 and maxValue < 0.5:
                lim = 0.5
            elif maxValue > 0.5 and maxValue < 1:
                lim = 2
            else:
                lim = 5

            # Plotting statokinesiogram
            with t1:
                plt.figure(figsize=(5, 5))
                plt.rcParams.update({'font.size': 12})
                if checkbox_1 == True:
                    plt.plot(yML, yAP, 'grey')
                if checkbox_2 == True:
                    plt.plot(yML[slider_min:slider_max],
                             yAP[slider_min:slider_max], 'k')
                plt.plot(ellipse_fit[:, 0], ellipse_fit[:, 1], 'r')
                plt.fill(ellipse_fit[:, 0], ellipse_fit[:,
                         1], color='tomato', alpha=0.5)
                plt.xlabel('Aceleração ML (g)')
                plt.ylabel('Aceleração AP (g)')
                plt.ylim(-limite, limite)
                plt.xlim(-limite, limite)
                bufferplot1 = BytesIO()
                plt.savefig(bufferplot1, format="jpg")
                bufferplot1.seek(0)
                st.pyplot(plt)

                plt.figure(figsize=(5, 5))
                plt.plot(frequencies, spectrum_amplitude_AP, 'k')
                plt.xlabel('Frequência Temporal (Hz)')
                plt.ylabel('Magnitude de aceleração AP (g)')
                plt.xlim(0, 6)
                plt.ylim(0, limite*0.05)
                bufferplot2 = BytesIO()
                plt.savefig(bufferplot2, format="png")
                bufferplot2.seek(0)
                st.pyplot(plt)

                # Stabilograms plot
            with t2:
                plt.figure(figsize=(5, 1.9))
                plt.rcParams.update({'font.size': 12})
                if checkbox_1 == True:
                    plt.plot(xAP, yAP, 'grey')
                if checkbox_2 == True:
                    plt.plot(xAP[slider_min:slider_max],
                             yAP[slider_min:slider_max], 'k')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração AP (g)')
                plt.ylim(-limite, limite)
                bufferplot3 = BytesIO()
                plt.savefig(bufferplot3, format="png")
                bufferplot3.seek(0)
                st.pyplot(plt)

                plt.figure(figsize=(5, 1.9))
                if checkbox_1 == True:
                    plt.plot(xML, yML, 'grey')
                if checkbox_2 == True:
                    plt.plot(xML[slider_min:slider_max],
                             yML[slider_min:slider_max], 'k')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração ML (g)')
                plt.ylim(-limite, limite)
                bufferplot4 = BytesIO()
                plt.savefig(bufferplot4, format="png")
                bufferplot4.seek(0)
                st.pyplot(plt)

                plt.figure(figsize=(5, 5))
                plt.plot(frequencies, spectrum_amplitude_ML, 'k')
                plt.xlabel('Frequência Temporal (Hz)')
                plt.ylabel('Energia da aceleração ML (g^2)')
                plt.xlim(0, 6)
                plt.ylim(0, limite*0.05)
                bufferplot5 = BytesIO()
                plt.savefig(bufferplot5, format="png")
                bufferplot5.seek(0)
                st.pyplot(plt)

                # Printing of the features values
            with t3:
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
                st.text('RMS AP (g) = ' + str(round(rmsAP, 5)))
                st.text('RMS ML (g) = ' + str(round(rmsML, 5)))
                st.text('Desvio total (g) = ' +
                        str(round(total_deviation, 3)))
                st.text('Área (g^2) = ' + str(round(area_value, 5)))
                st.text('Eixo maior (g) = ' + str(round(major_axis_value, 5)))
                st.text('Eixo menor (g) = ' + str(round(minor_axis_value, 5)))
                st.text('Ângulo de rotação (graus) = ' +
                        str(round(angle_deg_value, 2)))
                st.text('Frequência mediana AP (Hz) = ' +
                        str(round(median_frequency_AP, 4)))
                st.text('Frequência mediana ML (Hz) = ' +
                        str(round(median_frequency_ML, 4)))
                st.text('Energia das frequências baixas AP (g^2) = ' +
                        str(round(LF_energy_AP, 4)))
                st.text('Energia das frequências médias AP (g^2) = ' +
                        str(round(MF_energy_AP, 4)))
                st.text('Energia das frequências altas AP (g^2) = ' +
                        str(round(HF_energy_AP, 4)))
                st.text('Energia das frequências baixas ML (g^2) = ' +
                        str(round(LF_energy_ML, 4)))
                st.text('Energia das frequências médias ML (g^2) = ' +
                        str(round(MF_energy_ML, 4)))
                st.text('Energia das frequências altas ML (g^2) = ' +
                        str(round(HF_energy_ML, 4)))

                buffer = BytesIO()
                pdf = canvas.Canvas(buffer)

                # criando a linha
                pdf.setStrokeColorRGB(0, 0, 1)
                x1, y1 = 0, 769
                x2, y2 = 700, 769
                pdf.line(x1, y1, x2, y2)

                # criando o retângulo
                pdf.setFillColorRGB(0.8, 0.8, 1)  # Blue in RGB format
                pdf.setStrokeColorRGB(0, 0, 0, 0)  # Transparent in RGB format
                x1, y1 = 0, 0
                x2, y2 = 150, 900
                pdf.rect(x1, y1, x2 - x1, y2 - y1, fill=1)

                # criando texto de cabeçalho
                pdf.setFillColorRGB(0, 0, 1)  # Blue in RGB format
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(5, 800, "Universidade Federal do Pará")
                pdf.setFont("Helvetica", 8)
                pdf.drawString(5, 790, "Núcleo de Medicina Tropical")
                pdf.drawString(5, 780, "Laboratório de Neurologia Tropical")
                pdf.drawString(5, 770, "Atendimento em Saúde Digital")
                pdf.drawString(5, 720, "Smartphone: Xiaomi Mi 11 lite")
                pdf.drawString(5, 710, "Registro em L5")
                pdf.drawString(5, 700, f"Condição de {test_condition}")
                pdf.drawString(5, 690, f"Apoio {support}")
                pdf.drawString(5, 680, f"Duração de {duration} segundos")

                pdf.drawString(5, 100, "Núcleo de Medicina Tropical")
                pdf.drawString(5, 90, "Av. Generalíssimo Deodoro 92")
                pdf.drawString(5, 80, "Umarizal, Belém, Pará")
                pdf.drawString(5, 70, "66055240, Brasil")
                pdf.drawString(5, 60, "givagosouza@ufpa.br")
                pdf.drawString(5, 50, "91982653131")
                pdf.drawString(480, 20, "Givago da Silva Souza")
                pdf.drawString(470, 10, "Universidade Federal do Pará")

                pdf.setFont("Helvetica-Bold", 14)
                pdf.drawString(
                    200, 780, "RELATÓRIO DE AVALIAÇÃO DO EQUILÍBRIO ESTÁTICO")
                pdf.setFont("Helvetica-Bold", 8)
                pdf.drawString(155, 760, "Nome:")
                pdf.drawString(155, 745, "Data do teste:")
                pdf.drawString(155, 730, "Encaminhamento:")
                pdf.drawString(400, 760, "Data de nascimento:")
                pdf.drawString(400, 745, "Sexo:")
                pdf.drawString(400, 730, "Contato:")

                pdf.setFillColorRGB(0, 0, 0)
                pdf.setFont("Helvetica", 8)
                pdf.drawString(185, 760, name_participant)
                pdf.drawString(210, 745, str(date_participant))
                pdf.drawString(230, 730, doctor_participant)
                pdf.drawString(480, 760, str(birthdate_participant))
                pdf.drawString(425, 745, sex_participant)
                pdf.drawString(440, 730, contact_participant)

                pdf.setFillColorRGB(0, 0, 1)
                pdf.setFont("Helvetica-Bold", 12)
                pdf.drawString(160, 700, "Resultados:")

                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(160, 680, "Parâmetros estabilométricos:")

                pdf.setFont("Helvetica", 8)
                pdf.setFillColorRGB(0, 0, 0)  # Blue in RGB format
                pdf.drawString(165, 660, 'RMS AP (g) = ' +
                               str(round(rmsAP, 5)))
                pdf.drawString(165, 640, 'RMS ML (g) = ' +
                               str(round(rmsML, 5)))

                pdf.setFillColorRGB(0, 0, 1)
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(160, 610, "Parâmetros estatocinesiométricos:")

                pdf.setFillColorRGB(0, 0, 0)
                pdf.setFont("Helvetica", 8)
                pdf.drawString(165, 590, 'Desvio total (g) = ' +
                               str(round(total_deviation, 3)))
                pdf.drawString(165, 570, 'Área (g^2) = ' +
                               str(round(area_value, 5)))
                pdf.drawString(165, 550, 'Eixo maior (g) = ' +
                               str(round(major_axis_value, 5)))
                pdf.drawString(165, 530, 'Eixo menor (g) = ' +
                               str(round(minor_axis_value, 5)))
                pdf.drawString(165, 510, 'Ângulo de rotação (graus) = ' +
                               str(round(angle_deg_value, 2)))

                pdf.setFillColorRGB(0, 0, 1)  # Blue in RGB format
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(400, 680, "Parâmetros espectrais:")
                pdf.setFillColorRGB(0, 0, 0)
                pdf.setFont("Helvetica", 8)
                pdf.drawString(405, 660, 'Frequência mediana AP (Hz) = ' +
                               str(round(median_frequency_AP, 2)))
                pdf.drawString(405, 640, 'Frequência mediana ML (Hz) = ' +
                               str(round(median_frequency_ML, 2)))
                pdf.drawString(
                    405, 620, 'Energia das frequências baixas AP (g^2) = ' + str(round(LF_energy_AP, 2)))
                pdf.drawString(
                    405, 600, 'Energia das frequências médias AP (g^2) = ' + str(round(MF_energy_AP, 2)))
                pdf.drawString(
                    405, 580, 'Energia das frequências altas AP (g^2) = ' + str(round(HF_energy_AP, 2)))
                pdf.drawString(
                    405, 560, 'Energia das frequências baixas ML (g^2) = ' + str(round(LF_energy_ML, 2)))
                pdf.drawString(
                    405, 540, 'Energia das frequências médias ML (g^2) = ' + str(round(MF_energy_ML, 2)))
                pdf.drawString(
                    405, 520, 'Energia das frequências altas ML (g^2) = ' + str(round(HF_energy_ML, 2)))

                pdf.drawImage(ImageReader(bufferplot1), 170,
                              300, width=200, height=200)
                pdf.drawImage(ImageReader(bufferplot3), 380,
                              410, width=200, height=70)
                pdf.drawImage(ImageReader(bufferplot4), 380,
                              330, width=200, height=70)
                pdf.drawImage(ImageReader(bufferplot2), 170,
                              90, width=200, height=200)
                pdf.drawImage(ImageReader(bufferplot5), 380,
                              90, width=200, height=200)

                pdf.saveState()  # Salvar o estado atual
                pdf.rotate(90)  # Rotacionar em 90 graus
                # Posição x, y para o texto
                pdf.drawString(370, -165, "Aceleração AP (g)")
                # Posição x, y para o texto
                pdf.drawString(160, -165, "Energia AP (g^2)")
                pdf.drawString(160, -370, "Energia ML (g^2)")
                pdf.restoreState()

                pdf.drawString(470, 315, "Tempo (s)")

                # Save the PDF
                pdf.save()
                buffer.seek(0)
                st.download_button(
                    "Gerar relatório - Equilíbrio estático", buffer)
                # Nome do arquivo de texto de saída
                output_file = "output.txt"

                # Abrir o arquivo para escrita
                with open(output_file, "w") as file:
                    file.write('RMS AP (g) = ' + str(round(rmsAP, 5)) + "\n")
                    file.write('RMS ML (g) = ' + str(round(rmsML, 5)) + "\n")
                    file.write('Desvio total (g) = ' +
                               str(round(total_deviation, 3)) + "\n")
                    file.write('Área (g^2) = ' +
                               str(round(area_value, 5)) + "\n")
                    file.write('Eixo maior (g) = ' +
                               str(round(major_axis_value, 5)) + "\n")
                    file.write('Eixo menor (g) = ' +
                               str(round(minor_axis_value, 5)) + "\n")
                    file.write('Ângulo de rotação (graus) = ' +
                               str(round(angle_deg_value, 2)) + "\n")
                    file.write('Frequência mediana AP (Hz) = ' +
                               str(round(median_frequency_AP, 4)) + "\n")
                    file.write('Frequência mediana ML (Hz) = ' +
                               str(round(median_frequency_ML, 4)) + "\n")
                    file.write('Energia das frequências baixas AP (g^2) = ' +
                               str(round(LF_energy_AP, 4)) + "\n")
                    file.write('Energia das frequências médias AP (g^2) = ' +
                               str(round(MF_energy_AP, 4)) + "\n")
                    file.write('Energia das frequências altas AP (g^2) = ' +
                               str(round(HF_energy_AP, 4)) + "\n")
                    file.write('Energia das frequências baixas ML (g^2) = ' +
                               str(round(LF_energy_ML, 4)) + "\n")
                    file.write('Energia das frequências médias ML (g^2) = ' +
                               str(round(MF_energy_ML, 4)) + "\n")
                    file.write('Energia das frequências altas ML (g^2) = ' +
                               str(round(HF_energy_ML, 4)) + "\n")

                with open(output_file, "r") as file:
                    file_contents = file.read()

                # Usar st.download_button para baixar o arquivo
                st.download_button("Baixar resultados - Equilíbrio",
                                   data=file_contents, key='download_results')

                # Nome do arquivo de texto de saída
                output2_file = "output2.txt"
                x_coordinates = frequencies
                y_coordinates = spectrum_amplitude_AP
                z_coordinates = spectrum_amplitude_ML
                data = list(zip(x_coordinates, y_coordinates, z_coordinates))
                # Abrir o arquivo para escrita
                with open(output2_file, "w") as file:
                    file.write("X\tY\tZ\n")
                    for row in data:
                        file.write(f'{row[0]}\t{row[1]}\t{row[2]}\n')

                with open(output2_file, "r") as file:
                    file2_contents = file.read()

                st.download_button("Baixar resultados - Spectrum",
                                   data=file2_contents, key='download2_results')

# Algorithm to proceed iTUG analysis. All the analysis we’ve done in the iTUG was based on finding six transient events related to different stages of the task. (i) Task onset; (ii) Turn to return; (iii) Turn to sit; (iv) Sit-to-stand transition; (v) Stand-to-sit transition; and (vi) Task offset
with tab2:
    st.markdown(
        '<h2 style="text-align: center; color: blue;">Timed Up and Go test instrumentado</h2>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns([0.75, 1.75, 1])

    # Create acceleration and gyroscope file upload buttons
    uploaded_acc_iTUG = st.file_uploader(
        "Carregue o arquivo de texto do acelerômetro", type=["txt"],)
    uploaded_gyro_iTUG = st.file_uploader(
        "Carregue o arquivo de texto do giroscópio", type=["txt"],)

    if uploaded_acc_iTUG is not None:
        # Allocation of the acceleration data to the variables
        if uploaded_acc_iTUG is not None:
            custom_separator = ';'
            df = pd.read_csv(uploaded_acc_iTUG, sep=custom_separator)
            t = df.iloc[:, 0]
            x = df.iloc[:, 1]
            y = df.iloc[:, 2]
            z = df.iloc[:, 3]
            time = t

            # Pre-processing data: All channels were detrended, normalized to gravity acceleration, and interpolated to 100 Hz
            if np.max(x) > 9 or np.max(y) > 9 or np.max(z) > 9:
                x = signal.detrend(x/9.81)
                y = signal.detrend(y/9.81)
                z = signal.detrend(z/9.81)
            else:
                x = signal.detrend(x)
                y = signal.detrend(y)
                z = signal.detrend(z)
            interpf = scipy.interpolate.interp1d(time, x)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            x_ = interpf(time_)
            t, x = time_/1000, x_
            interpf = scipy.interpolate.interp1d(time, y)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            y_ = interpf(time_)
            t, y = time_/1000, y_
            interpf = scipy.interpolate.interp1d(time, z)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            z_ = interpf(time_)
            t, z = time_/1000, z_

            # Calculating acceleration data norm
            norm_waveform = np.sqrt(x**2+y**2+z**2)

            # Filtering acceleration data norm
            norm_waveform = butterworth_filter(
                norm_waveform, 4, 100, order=2, btype='low')

        # Allocation of the gyroscope data to the variables
        if uploaded_gyro_iTUG is not None:
            name_participant, date_participant, doctor_participant, birthdate_participant, sex_participant, contact_participant = individual_info_iTUG()

            custom_separator = ';'
            df_gyro = pd.read_csv(uploaded_gyro_iTUG, sep=custom_separator)
            t_gyro = df_gyro.iloc[:, 0]
            x_gyro = df_gyro.iloc[:, 1]
            y_gyro = df_gyro.iloc[:, 2]
            z_gyro = df_gyro.iloc[:, 3]
            time_gyro = t_gyro

            # Pre-processing data: All channels were detrended, and interpolated to 100 Hz
            x_gyro = signal.detrend(x_gyro)
            y_gyro = signal.detrend(y_gyro)
            z_gyro = signal.detrend(z_gyro)
            interpf = scipy.interpolate.interp1d(time_gyro, x_gyro)
            time_gyro_ = np.arange(
                start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
            x_gyro_ = interpf(time_gyro_)
            t_gyro, x_gyro = time_gyro_/1000, x_gyro_
            interpf = scipy.interpolate.interp1d(time_gyro, y_gyro)
            time_gyro_ = np.arange(
                start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
            y_gyro_ = interpf(time_gyro_)
            t_gyro, y_gyro = time_gyro_/1000, y_gyro_
            interpf = scipy.interpolate.interp1d(time_gyro, z_gyro)
            time_gyro_ = np.arange(
                start=time_gyro[0], stop=time_gyro[len(time_gyro)-1], step=10)
            z_gyro_ = interpf(time_gyro_)
            t_gyro, z_gyro = time_gyro_/1000, z_gyro_

            # Calculating norm for angular velocity
            norm_waveform_gyro = np.sqrt(x_gyro**2+y_gyro**2+z_gyro**2)

            # Filtering norm for acceleration
            norm_waveform_gyro = butterworth_filter(
                norm_waveform_gyro, 1.5, 100, order=2, btype='low')

            # Creating controls to interacts with the plots
            with t1:
                # Create slider widgets to set limits of baselines for the test onset and test offset
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Ajustar as baselines</h3>', unsafe_allow_html=True)
                length = len(norm_waveform_gyro)
                slider_baseline1 = st.number_input(
                    "Selecione o início do baseline do ONSET", min_value=1, max_value=length, step=1, value=50)
                slider_baseline2 = st.number_input(
                    "Selecione o final do baseline do ONSET", min_value=1, max_value=length, value=100, step=1)
                slider_baseline3 = st.number_input(
                    "Selecione o início do baseline do OFFSET", min_value=1, max_value=length, value=length-150, step=1)
                slider_baseline4 = st.number_input(
                    "Selecione o final do baseline do OFFSET", min_value=1, max_value=length, value=length-100, step=1)

                # We used the data from gyroscope to find the onset of the sitting to standing transition following Van Lummel et al. (2013) recommendation. We calculated the first derivative of the norm because we observed that it has less variability than the norm and it would facilitate to find the moment of the deflection from sitting to standing position. The basic idea to find the onset of the task was to choose a period during the pre-standing as baseline. Then, we calculated the average and standard deviation of the baseline from gyroscope first derivative vector and searched for the moment that the vector value exceeded the mean plus 4*standard deviation
                firstDerivative = np.diff(norm_waveform_gyro)

            # Setting the limits of the baseline to find the task onset. It is selected a range in the beginning of the recording
                if slider_baseline1 < slider_baseline2:
                    avg_firstderivative = np.mean(
                        firstDerivative[slider_baseline1:slider_baseline2])
                    std_firstderivative = np.std(
                        firstDerivative[slider_baseline1:slider_baseline2])
                    loc_onset = slider_baseline2

                # Finding the task onset
                for i in firstDerivative[slider_baseline2:length-slider_baseline2]:
                    if i < avg_firstderivative + 4 * std_firstderivative:
                        loc_onset = loc_onset + 1
                    else:
                        break

            # Setting the limits of the baseline to find the task onset. It is selected a range in the end of the recording.
                if slider_baseline3 < slider_baseline4:
                    avg_firstderivative_offset = np.mean(
                        firstDerivative[slider_baseline3:slider_baseline4])
                    std_firstderivative_offset = np.std(
                        firstDerivative[slider_baseline3:slider_baseline4])
                    loc_offset = slider_baseline3

                # Finding the task offset
                for i in reversed(firstDerivative[1:slider_baseline3]):
                    if i < avg_firstderivative_offset + 4*std_firstderivative_offset:
                        loc_offset = loc_offset - 1
                    else:
                        break

            # Setting the sliders with onset and offset positions
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Ajustes manuais</h3>', unsafe_allow_html=True)
                slider_onset = st.number_input(
                    "Ajuste o momento do ONSET", min_value=1, max_value=length, value=loc_onset, step=1)
                slider_offset = st.number_input(
                    "Ajuste o momento do OFFSET", min_value=1, max_value=length-1, value=loc_offset, step=1)

                # Next step is to find the angular velocity peak during the return turn and pre-sitting turn. For that, we found the amplitude peaks and the position of the two largest amplitudes in the gyroscope norm. To indicate which component is each amplitude we compared the location in the vector. The earlier is from the return turn and the later is from the pre-sitting turn
                peaks, _ = find_peaks(norm_waveform_gyro, height=0.25)
                amplitude = norm_waveform_gyro[peaks]
                amplitude = sorted(amplitude, reverse=True)
                a = 0
                for i in norm_waveform_gyro:
                    a = a + 1
                    if i == amplitude[0]:
                        loc1 = a
                        latency1 = t_gyro[a]
                        amplitude1 = norm_waveform_gyro[a]
                        break
                a = 0
                for i in norm_waveform_gyro:
                    a = a + 1
                    if i == amplitude[1]:
                        loc2 = a
                        latency2 = t_gyro[a]
                        amplitude2 = norm_waveform_gyro[a]
                        break
                if latency1 > latency2:
                    g1_latency = latency2
                    g1_amplitude = amplitude2
                    loc_g1 = loc2
                    g2_latency = latency1
                    g2_amplitude = amplitude1
                    loc_g2 = loc1
                else:
                    g1_latency = latency1
                    g1_amplitude = amplitude1
                    loc_g1 = loc1
                    g2_latency = latency2
                    g2_amplitude = amplitude2
                    loc_g2 = loc2

                # Setting the sliders with angular velocity peak positions
                slider_G1 = st.number_input(
                    "Selecionar o pico de G1", min_value=1, max_value=length-1, value=loc_g1, step=1)
                slider_G2 = st.number_input(
                    "Selecione o pico de G2", min_value=1, max_value=length-1, value=loc_g2, step=1)

                # Now, we search the peak in the acceleration norm between the task onset and 200 ms later. The value of 200 ms was arbitrary and we observed that was suitable to the peak detection. This peak is the acceleration peak during the sit-to-standing transition.
                standing_peak_acc = np.max(
                    norm_waveform[slider_onset:slider_onset+200])
                standing_peak_loc = 0
                for i in norm_waveform:
                    if i != standing_peak_acc:
                        standing_peak_loc = standing_peak_loc + 1
                        standing_peak_latency = t[standing_peak_loc]
                    else:
                        break

                # Now, we search the peak in the acceleration norm between moment of the angular velocity peak of the pre-sitting turn and the task offset. This peak is the acceleration peak during the standing-to-sit transition.
                sitting_peak_acc = np.max(norm_waveform[loc_g2:loc_offset])
                sitting_peak_loc = 0
                for i in norm_waveform:
                    if i != sitting_peak_acc:
                        sitting_peak_loc = sitting_peak_loc + 1
                        sitting_peak_latency = t[sitting_peak_loc]
                    else:
                        break

                # Setting the sliders with acceleration peak positions
                slider_A1 = st.number_input(
                    "Secione o pico de A1", min_value=1, max_value=length-1, value=standing_peak_loc, step=1)
                slider_A2 = st.number_input(
                    "Selecione o pico de A2", min_value=1, max_value=length-1, value=sitting_peak_loc, step=1)

                # Extracting the features from iTUG
                sit_to_standing_duration = t[slider_A1] - t_gyro[slider_onset]
                walking_to_go_duration = t_gyro[slider_G1] - t[slider_A1]
                walking_to_return_duration = t_gyro[slider_G2] - \
                    t_gyro[slider_G1]
                return_to_sit_duration = t[slider_A2] - t_gyro[slider_G2]
                standing_to_sit_duration = t_gyro[slider_offset] - t[slider_A2]
                total_duration = sit_to_standing_duration + walking_to_go_duration + \
                    walking_to_return_duration + return_to_sit_duration + standing_to_sit_duration

                # Creating arrays to plot the baselines for onset and offset detection
                shade_baseline1_x = [
                    t[slider_baseline1], t[slider_baseline1], t[slider_baseline2], t[slider_baseline2]]
                shade_baseline1_y = [0, 0.5, 0.5, 0]
                shade_baseline2_x = [
                    t[slider_baseline3], t[slider_baseline3], t[slider_baseline4], t[slider_baseline4]]
                shade_baseline2_y = [0, 0.5, 0.5, 0]
            with t2:
                # Plotting the gyroscope norm with iTUG stages in color shades
                plt.figure(figsize=(5, 3))
                plt.plot(t_gyro, norm_waveform_gyro, 'k')
                plt.fill(shade_baseline1_x, shade_baseline1_y, 'b', alpha=0.2)
                plt.fill(shade_baseline2_x, shade_baseline2_y, 'b', alpha=0.2)
                lim_y = np.max(norm_waveform_gyro)
                shade_sitting_2_standing_x = [
                    t_gyro[slider_onset], t_gyro[slider_onset], t[slider_A1], t[slider_A1]]
                shade_y = [0, lim_y, lim_y, 0]
                plt.fill(shade_sitting_2_standing_x, shade_y,
                         color=(1, 0.5, 0.5), alpha=0.65)
                shade_go_x = [t[slider_A1], t[slider_A1],
                              t_gyro[slider_G1], t_gyro[slider_G1]]
                plt.fill(shade_go_x, shade_y, color=(0.6, 1, 0.5), alpha=0.65)
                shade_return_x = [
                    t_gyro[slider_G1], t_gyro[slider_G1], t_gyro[slider_G2], t_gyro[slider_G2]]
                plt.fill(shade_return_x, shade_y,
                         color=(1, 1, 0.4), alpha=0.65)
                shade_pre_sitting_x = [
                    t_gyro[slider_G2], t_gyro[slider_G2], t[slider_A2], t[slider_A2]]
                plt.fill(shade_pre_sitting_x, shade_y,
                         color=(0.5, 0.6, 1), alpha=0.65)
                shade_sitting_x = [t[slider_A2], t[slider_A2],
                                   t_gyro[slider_offset], t_gyro[slider_offset]]
                plt.fill(shade_sitting_x, shade_y,
                         color=(0.4, 0.4, 0.4), alpha=0.65)
                baseline_duration = str(
                    round(t[slider_baseline2] - t[slider_baseline1], 2)) + " s"
                plt.text(t[slider_baseline1], 0.6, baseline_duration)
                baseline_duration_offset = str(
                    round(t[slider_baseline4] - t[slider_baseline3], 2)) + " s"
                plt.text(t[slider_baseline3], 0.6, baseline_duration_offset)
                plt.plot([t_gyro[slider_onset], t_gyro[slider_onset]], [
                         0, lim_y], '--r')
                plt.plot([t_gyro[slider_offset], t_gyro[slider_offset]], [
                         0, lim_y], '--b')
                plt.plot(t_gyro[slider_G1], norm_waveform_gyro[slider_G1], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t_gyro[slider_G1-20],
                         norm_waveform_gyro[slider_G1]*1.05, 'G1')
                plt.plot(t_gyro[slider_G2], norm_waveform_gyro[slider_G2], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t_gyro[slider_G2-20],
                         norm_waveform_gyro[slider_G2]*1.05, 'G2')
                plt.ylim(0, np.max(norm_waveform_gyro)*1.2)
                plt.xlabel('Tempo (s)')
                plt.ylabel('Velocidade angular (rad/s)')
                buffertug1 = BytesIO()
                plt.savefig(buffertug1, format="png")
                buffertug1.seek(0)
                st.pyplot(plt)

           # Plotting the accelerometer norm with iTUG stages in color shades
                fig = plt.figure(figsize=(5, 3))
                plt.plot(t, norm_waveform, 'k')
                lim_y = np.max(norm_waveform)
                shade_y = [0, lim_y, lim_y, 0]
                plt.plot(t[slider_A1], norm_waveform[slider_A1], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t[slider_A1-20], norm_waveform[slider_A1]*1.05, 'A1')
                plt.plot(t[slider_A2], norm_waveform[slider_A2], marker='o',
                         markerfacecolor='none', markeredgecolor='k', markersize=14)
                plt.text(t[slider_A2-20], norm_waveform[slider_A2]*1.05, 'A2')
                plt.fill(shade_sitting_2_standing_x, shade_y,
                         color=(1, 0.5, 0.5), alpha=0.65)
                plt.fill(shade_go_x, shade_y, color=(0.6, 1, 0.5), alpha=0.65)
                plt.fill(shade_return_x, shade_y,
                         color=(1, 1, 0.4), alpha=0.65)
                plt.fill(shade_pre_sitting_x, shade_y,
                         color=(0.5, 0.6, 1), alpha=0.65)
                plt.fill(shade_sitting_x, shade_y,
                         color=(0.4, 0.4, 0.4), alpha=0.65)
                plt.plot([t_gyro[slider_onset], t_gyro[slider_onset]], [
                         0, lim_y], '--r')
                plt.plot([t_gyro[slider_offset], t_gyro[slider_offset]], [
                         0, lim_y], '--b')
                plt.xlabel('Tempo (s)')
                plt.ylabel('Aceleração (g)')
                buffertug2 = BytesIO()
                plt.savefig(buffertug2, format="png")
                buffertug2.seek(0)
                st.pyplot(plt)
           # Priting the feature values
            with t3:
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
                st.text('Duração total (s) = ' +
                        str(round(total_duration, 2)))
                st.text('Duração de sentar para levantar (s) = ' +
                        str(round(sit_to_standing_duration, 2)))
                st.text('Duração da caminhada de ida (s) = ' +
                        str(round(walking_to_go_duration, 2)))
                st.text('Duração da caminhada de retorno (s) = ' +
                        str(round(walking_to_return_duration, 2)))
                st.text('Duração de em pé para sentar (s) = ' +
                        str(round(standing_to_sit_duration, 2)))
                st.text('Pico de A1 (g) = ' +
                        str(round(norm_waveform[slider_A1], 2)))
                st.text('Pico de A2 (g) = ' +
                        str(round(norm_waveform[slider_A2], 2)))
                st.text('Pico de G1 (rad/s) = ' +
                        str(round(norm_waveform_gyro[slider_G1], 2)))
                st.text('Pico de G2 (rad/s) = ' +
                        str(round(norm_waveform_gyro[slider_G2], 2)))

                buffer = BytesIO()
                pdf = canvas.Canvas(buffer)

                # criando a linha
                pdf.setStrokeColorRGB(0, 0, 1)
                x1, y1 = 0, 769
                x2, y2 = 700, 769
                pdf.line(x1, y1, x2, y2)

                # criando o retângulo
                pdf.setFillColorRGB(0.8, 0.8, 1)  # Blue in RGB format
                pdf.setStrokeColorRGB(0, 0, 0, 0)  # Transparent in RGB format
                x1, y1 = 0, 0
                x2, y2 = 150, 900
                pdf.rect(x1, y1, x2 - x1, y2 - y1, fill=1)

                # criando texto de cabeçalho
                pdf.setFillColorRGB(0, 0, 1)  # Blue in RGB format
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(5, 800, "Universidade Federal do Pará")
                pdf.setFont("Helvetica", 8)
                pdf.drawString(5, 790, "Núcleo de Medicina Tropical")
                pdf.drawString(5, 780, "Laboratório de Neurologia Tropical")
                pdf.drawString(5, 770, "Atendimento em Saúde Digital")
                pdf.drawString(5, 720, "Smartphone: Xiaomi Mi 11 lite")
                pdf.drawString(5, 710, "Registro em L5")
                pdf.drawString(5, 700, "Timed Up and Go test instrumentado")

                pdf.drawString(5, 100, "Núcleo de Medicina Tropical")
                pdf.drawString(5, 90, "Av. Generalíssimo Deodoro 92")
                pdf.drawString(5, 80, "Umarizal, Belém, Pará")
                pdf.drawString(5, 70, "66055240, Brasil")
                pdf.drawString(5, 60, "givagosouza@ufpa.br")
                pdf.drawString(5, 50, "91982653131")
                pdf.drawString(480, 20, "Givago da Silva Souza")
                pdf.drawString(470, 10, "Universidade Federal do Pará")

                pdf.setFont("Helvetica-Bold", 14)
                pdf.drawString(
                    200, 780, "RELATÓRIO DE AVALIAÇÃO DA MOBILIDADE")
                pdf.setFont("Helvetica-Bold", 8)
                pdf.drawString(155, 760, "Nome:")
                pdf.drawString(155, 745, "Data do teste:")
                pdf.drawString(155, 730, "Encaminhamento:")
                pdf.drawString(400, 760, "Data de nascimento:")
                pdf.drawString(400, 745, "Sexo:")
                pdf.drawString(400, 730, "Contato:")

                pdf.setFillColorRGB(0, 0, 0)
                pdf.setFont("Helvetica", 8)
                pdf.drawString(185, 760, name_participant)
                pdf.drawString(210, 745, str(date_participant))
                pdf.drawString(230, 730, doctor_participant)
                pdf.drawString(480, 760, str(birthdate_participant))
                pdf.drawString(425, 745, sex_participant)
                pdf.drawString(440, 730, contact_participant)

                pdf.setFillColorRGB(0, 0, 1)
                pdf.setFont("Helvetica-Bold", 12)
                pdf.drawString(160, 700, "Resultados:")

                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(160, 680, "Parâmetros quantitativos:")

                pdf.setFont("Helvetica", 8)
                pdf.setFillColorRGB(0, 0, 0)
                pdf.drawString(165, 660, 'Duração total (s) = ' +
                               str(round(total_duration, 2)))
                pdf.drawString(165, 640, 'Duração de sentar para levantar (s) = ' +
                               str(round(sit_to_standing_duration, 2)))
                pdf.drawString(165, 620, 'Duração da caminhada de ida (s) = ' +
                               str(round(walking_to_go_duration, 2)))
                pdf.drawString(165, 600, 'Duração da caminhada de retorno (s) = ' +
                               str(round(walking_to_return_duration, 2)))
                pdf.drawString(165, 580, 'Duração da caminhada de retorno (s) = ' +
                               str(round(standing_to_sit_duration, 2)))
                pdf.drawString(165, 560, 'Pico de A1 (g) = ' +
                               str(round(norm_waveform[slider_A1], 2)))
                pdf.drawString(165, 540, 'Pico de A2 (g) = ' +
                               str(round(norm_waveform[slider_A2], 2)))
                pdf.drawString(165, 520, 'Pico de G1 (rad/s) = ' +
                               str(round(norm_waveform_gyro[slider_G1], 2)))
                pdf.drawString(165, 500, 'Pico de G2 (rad/s) = ' +
                               str(round(norm_waveform_gyro[slider_G2], 2)))

                pdf.drawImage(ImageReader(buffertug1), 170,
                              300, width=220, height=140)
                pdf.drawImage(ImageReader(buffertug2), 380,
                              300, width=220, height=140)

                pdf.drawString(250, 290, "Tempo (s)")
                pdf.drawString(470, 290, "Tempo (s)")

                # Save the PDF
                pdf.save()
                buffer.seek(0)
                st.download_button("Criar relatório - Mobilidade", buffer)
                # Nome do arquivo de texto de saída
                output_file = "output.txt"

                # Abrir o arquivo para escrita
                with open(output_file, "w") as file:
                    file.write('Duração total (s) = ' +
                               str(round(total_duration, 2)) + "\n")
                    file.write('Duração de sentar para levantar (s) = ' +
                               str(round(sit_to_standing_duration, 2)) + "\n")
                    file.write('Duração da caminhada de ida (s) = ' +
                               str(round(walking_to_go_duration, 2)) + "\n")
                    file.write('Duração da caminhada de retorno (s) = ' +
                               str(round(walking_to_return_duration, 2)) + "\n")
                    file.write('Duração de pé para sentar (s) = ' +
                               str(round(standing_to_sit_duration, 2)) + "\n")
                    file.write('Pico de A1 (g) = ' +
                               str(round(norm_waveform[slider_A1], 2)) + "\n")
                    file.write('Pico de A2 (g) = ' +
                               str(round(norm_waveform[slider_A2], 2)) + "\n")
                    file.write('Pico de G1 (rad/s) = ' +
                               str(round(norm_waveform_gyro[slider_G1], 2)) + "\n")
                    file.write('Pico de G2 (rad/s) = ' +
                               str(round(norm_waveform_gyro[slider_G2], 2)) + "\n")

                with open(output_file, "r") as file:
                    file_contents = file.read()

                # Usar st.download_button para baixar o arquivo
                st.download_button("Baixar resultados - Mobilidade",
                                   data=file_contents, key='download_TUG_results')
with tab3:
    st.markdown(
        '<h2 style="text-align: center; color: blue;">Finger tapping test realizado em smartphone</h2>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns([1, 1.75, 1])
    uploaded_ftt = st.file_uploader(
        "Selecione o arquivo de texto do FFT", type=["txt"],)

    if uploaded_ftt is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_ftt.read())
            st.write("Arquivo escolhido:", temp_file.name)
           # Define the file pathc
        file_path = temp_file.name
        # Create a dictionary to store the data
        name_participant, date_participant, doctor_participant, birthdate_participant, sex_participant, contact_participant = individual_info_FTT()
        options = ["Direita", "Esquerda"]
        hand = st.selectbox("Mão testada:", options)
        data = {}

        # Define a regular expression pattern to match key-value pairs
        pattern = r'([^:]+):\s*(.+)'

        # Open the file and read its content
        with open(file_path, "r") as file:
            for line in file:
                match = re.match(pattern, line)
                if match:
                    key, value = match.groups()
                    data[key] = value

        # Print the extracted data

        x_lim = float(data['Width'])
        y_lim = float(data['Height'])
        skip_rows = 9
        csv_data = pd.read_csv(file_path, skiprows=skip_rows)
        t = csv_data.iloc[:, 0]/1000
        intervals = np.diff(t)
        coord_x = csv_data.iloc[:, 1]
        coord_y = csv_data.iloc[:, 2]
        coord_x = coord_x.astype(float)
        coord_y = coord_y.astype(float)
        field = csv_data.iloc[:, 3]
        mat_x = [0, 0, x_lim, x_lim]
        mat_y = [0, y_lim, y_lim, 0]

        dados = coord_x, coord_y

        ellipse_x, ellipse_y, D, d, angle = ellipse_model(coord_x, coord_y)

        n_touches = len(field)
        n_errors = 0
        for i in field:
            if i == 0:
                n_errors = n_errors + 1
        mean_interval = np.mean(intervals)
        max_interval = np.max(intervals)
        min_interval = np.min(intervals)
        std_interval = np.std(intervals)
        ellipse_area = np.pi*D*d
        ellipse_major_axis = D
        ellipse_minor_axis = d
        ellipse_rotate_angle = angle
        total_deviation_ftt = np.sum(np.sqrt(coord_x**2+coord_y**2))

        with t1:
            plt.figure(figsize=(5, 5))
            plt.fill(mat_x, mat_y, 'k', alpha=0.5)
            plt.plot(coord_x, coord_y, '+', markersize=1, markeredgecolor='k')
            plt.plot(ellipse_x, ellipse_y, 'r')
            plt.fill(ellipse_x, ellipse_y, 'r', alpha=0.3)
            plt.xlim(0, y_lim)
            plt.ylim(0, y_lim)
            plt.axis('off')
            plt.gca().set_aspect('equal')
            bufferftt1 = BytesIO()
            plt.savefig(bufferftt1, format="png")
            bufferftt1.seek(0)
            st.pyplot(plt)
        with t2:
            plt.figure(figsize=(5, 5))
            plt.plot(t[0:len(intervals)], intervals, 'g')
            plt.xlim(0, 30)
            plt.ylim(0, np.max(intervals)*1.25)
            plt.xlabel('Tempo (s)')
            plt.ylabel('Intervalo entre os toques (s)')
            bufferftt2 = BytesIO()
            plt.savefig(bufferftt2, format="png")
            bufferftt2.seek(0)
            st.pyplot(plt)
        with t3:
            st.markdown(
                '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
            st.markdown("*Parâmetros globais*")
            st.text('Número de toques = ' + str(n_touches))
            st.text('Número de erros = ' + str(n_errors))
            st.markdown("*Parâmetros temporais*")
            st.text('Intervalo médio (s) = ' + str(round(mean_interval, 3)))
            st.text('Desvio-padrão dos intervalos (s) = ' +
                    str(round(std_interval, 3)))
            st.text('Intervalo máximo (s) = ' + str(round(max_interval, 2)))
            st.text('Intervalo mínimo (s) = ' + str(round(min_interval, 2)))
            st.markdown("*Parâmetros espaciais*")
            st.text('Desvio total (px) = ' +
                    str(round(total_deviation_ftt, 2)))
            st.text('Área da elipse (px)= ' + str(round(ellipse_area, 2)))
            st.text('Eixo maior (px) = ' +
                    str(round(ellipse_major_axis, 2)))
            st.text('Eixo menor (px) = ' +
                    str(round(ellipse_minor_axis, 2)))
            st.text('Ângulo de rotação (graus) = ' +
                    str(round(ellipse_rotate_angle, 2)))
            buffer = BytesIO()

            pdf = canvas.Canvas(buffer)

            # criando a linha
            pdf.setStrokeColorRGB(0, 0, 1)
            x1, y1 = 0, 769
            x2, y2 = 700, 769
            pdf.line(x1, y1, x2, y2)

            # criando o retângulo
            pdf.setFillColorRGB(0.8, 0.8, 1)  # Blue in RGB format
            pdf.setStrokeColorRGB(0, 0, 0, 0)  # Transparent in RGB format
            x1, y1 = 0, 0
            x2, y2 = 150, 900
            pdf.rect(x1, y1, x2 - x1, y2 - y1, fill=1)

            # criando texto de cabeçalho
            pdf.setFillColorRGB(0, 0, 1)  # Blue in RGB format
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(5, 800, "Universidade Federal do Pará")
            pdf.setFont("Helvetica", 8)
            pdf.drawString(5, 790, "Núcleo de Medicina Tropical")
            pdf.drawString(5, 780, "Laboratório de Neurologia Tropical")
            pdf.drawString(5, 770, "Atendimento em Saúde Digital")
            pdf.drawString(5, 720, "Smartphone: Xiaomi Mi 11 lite")
            pdf.drawString(5, 710, f"Mão testada: {hand}")
            pdf.drawString(5, 700, "Finger Tapping Test para alvo central")

            pdf.drawString(5, 100, "Núcleo de Medicina Tropical")
            pdf.drawString(5, 90, "Av. Generalíssimo Deodoro 92")
            pdf.drawString(5, 80, "Umarizal, Belém, Pará")
            pdf.drawString(5, 70, "66055240, Brasil")
            pdf.drawString(5, 60, "givagosouza@ufpa.br")
            pdf.drawString(5, 50, "91982653131")
            pdf.drawString(480, 20, "Givago da Silva Souza")
            pdf.drawString(470, 10, "Universidade Federal do Pará")

            pdf.setFillColorRGB(0, 0, 1)  # Blue in RGB format
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(
                170, 780, "RELATÓRIO DE AVALIAÇÃO DA COORDENAÇÃO MOTORA")
            pdf.setFont("Helvetica-Bold", 8)
            pdf.drawString(155, 760, "Nome:")
            pdf.drawString(155, 745, "Data do teste:")
            pdf.drawString(155, 730, "Encaminhamento:")
            pdf.drawString(400, 760, "Data de nascimento:")
            pdf.drawString(400, 745, "Sexo:")
            pdf.drawString(400, 730, "Contato:")

            pdf.setFillColorRGB(0, 0, 0)
            pdf.setFont("Helvetica", 8)
            pdf.drawString(185, 760, name_participant)
            pdf.drawString(210, 745, str(date_participant))
            pdf.drawString(230, 730, doctor_participant)
            pdf.drawString(480, 760, str(birthdate_participant))
            pdf.drawString(425, 745, sex_participant)
            pdf.drawString(440, 730, contact_participant)

            pdf.setFillColorRGB(0, 0, 1)
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(160, 700, "Resultados:")

            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(160, 680, "Parâmetros globais:")

            pdf.setFont("Helvetica", 8)
            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawString(165, 660, 'Número de toques = ' + str(n_touches))
            pdf.drawString(165, 640, 'Número de erros = ' + str(n_errors))

            pdf.setFillColorRGB(0, 0, 1)
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(160, 620, "Parâmetros temporais:")

            pdf.setFont("Helvetica", 8)
            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawString(165, 600, 'Intervalo médio (s) = ' +
                           str(round(mean_interval, 3)))
            pdf.drawString(165, 580, 'Desvio-padrão dos intervalos (s) = ' +
                           str(round(std_interval, 3)))
            pdf.drawString(165, 560, 'Intervalo máximo (s) = ' +
                           str(round(max_interval, 2)))
            pdf.drawString(165, 540, 'Intervalo mínimo (s) = ' +
                           str(round(min_interval, 2)))

            pdf.setFillColorRGB(0, 0, 1)
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(400, 680, "Parâmetros espaciais:")

            pdf.setFont("Helvetica", 8)
            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawString(405, 660, 'Desvio total (px) = ' +
                           str(round(total_deviation_ftt, 2)))
            pdf.drawString(405, 640, 'Área da elipse (px)= ' +
                           str(round(ellipse_area, 2)))
            pdf.drawString(405, 620, 'Eixo maior (px) = ' +
                           str(round(ellipse_major_axis, 2)))
            pdf.drawString(405, 600, 'Eixo menor (px) = ' +
                           str(round(ellipse_minor_axis, 2)))
            pdf.drawString(405, 580, 'Ângulo de rotação (graus) = ' +
                           str(round(ellipse_rotate_angle, 2)))

            pdf.drawImage(ImageReader(bufferftt1), 170,
                          150, width=270, height=300)
            pdf.drawImage(ImageReader(bufferftt2), 380,
                          200, width=220, height=200)

            # Save the PDF
            pdf.save()
            buffer.seek(0)
            st.download_button("Criar relatório - Coordenação motora", buffer)
            # Nome do arquivo de texto de saída
            output_file = "output.txt"

            # Abrir o arquivo para escrita
            with open(output_file, "w") as file:
                file.write('Número de toques = ' + str(n_touches) + "\n")
                file.write('Número de erros = ' + str(n_errors) + "\n")
                file.write('Intervalo médio (s) = ' +
                           str(round(mean_interval, 3)) + "\n")
                file.write('Desvio-padrão dos intervalos (s) = ' +
                           str(round(std_interval, 3)) + "\n")
                file.write('Intervalo máximo (s) = ' +
                           str(round(max_interval, 2)) + "\n")
                file.write('Intervalo mínimo (s) = ' +
                           str(round(min_interval, 2)) + "\n")
                file.write('Desvio total (px) = ' +
                           str(round(total_deviation_ftt, 2)) + "\n")
                file.write('Área da elipse (px)= ' +
                           str(round(ellipse_area, 2)) + "\n")
                file.write('Eixo maior (px) = ' +
                           str(round(ellipse_major_axis, 2)) + "\n")
                file.write('Eixo menor (px) = ' +
                           str(round(ellipse_minor_axis, 2)) + "\n")
                file.write('Ângulo de rotação (graus) = ' +
                           str(round(ellipse_rotate_angle, 2)) + "\n")
            with open(output_file, "r") as file:
                file_contents = file.read()

            # Usar st.download_button para baixar o arquivo
            st.download_button("Baixar resultados - Coordenação motora",
                               data=file_contents, key='download_FTT_results')
with tab4:
    st.markdown(
        '<h2 style="text-align: center; color: blue;">Avaliação do tremor de mão por smartphone</h2>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns([1, 1, 0.7])

    # Create acceleration and gyroscope file upload buttons
    uploaded_acc_tremor = st.file_uploader(
        "Selecione o arquivo de texto do tremor", type=["txt"],)
    name_participant, date_participant, doctor_participant, birthdate_participant, sex_participant, contact_participant = individual_info_tremor()
    options = ["Direita", "Esquerda"]
    hand = st.selectbox("Mão avaliada:", options)
    options = ["em repouso", "postural"]
    condition = st.selectbox("Condição de teste:", options)
    if uploaded_acc_tremor is not None:
        # Allocation of the acceleration data to the variables
        if uploaded_acc_tremor is not None:
            custom_separator = ';'
            df = pd.read_csv(uploaded_acc_tremor, sep=custom_separator)
            t = df.iloc[:, 0]
            x = df.iloc[:, 1]
            y = df.iloc[:, 2]
            z = df.iloc[:, 3]
            time = t

            # Pre-processing data: All channels were detrended, normalized to gravity acceleration, and interpolated to 100 Hz
            if np.max(x) > 9 or np.max(y) > 9 or np.max(z) > 9:
                x = signal.detrend(x/9.81)
                y = signal.detrend(y/9.81)
                z = signal.detrend(z/9.81)
            else:
                x = signal.detrend(x)
                y = signal.detrend(y)
                z = signal.detrend(z)
            interpf = scipy.interpolate.interp1d(time, x)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            x_ = interpf(time_)
            t, x = time_/1000, x_
            interpf = scipy.interpolate.interp1d(time, y)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            y_ = interpf(time_)
            t, y = time_/1000, y_
            interpf = scipy.interpolate.interp1d(time, z)
            time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
            z_ = interpf(time_)
            t, z = time_/1000, z_

            # Calculating acceleration data norm
            norm_waveform_tremor = np.sqrt(x**2+y**2+z**2)

            # Filtering acceleration data norm
            # norm_waveform_tremor = butterworth_filter(
            # norm_waveform_tremor, 10, 100, order=2, btype='low')

            p_spectrum_x_1, t_freq_x_1 = tremor_fft(x[0:499])
            p_spectrum_y_1, t_freq_y_1 = tremor_fft(y[0:499])
            p_spectrum_z_1, t_freq_z_1 = tremor_fft(z[0:499])
            p_spectrum_1_norm = np.sqrt(
                p_spectrum_x_1**2+p_spectrum_y_1**2+p_spectrum_z_1**2)

            max_sweep_1 = np.max([x[0:499], y[0:499], z[0:499]])
            min_sweep_1 = np.min([x[0:499], y[0:499], z[0:499]])
            max_power_1 = np.max(
                [p_spectrum_x_1, p_spectrum_y_1, p_spectrum_z_1])

            p_spectrum_x_2, t_freq_x_2 = tremor_fft(x[500:999])
            p_spectrum_y_2, t_freq_y_2 = tremor_fft(y[500:999])
            p_spectrum_z_2, t_freq_z_2 = tremor_fft(z[500:999])
            p_spectrum_2_norm = np.sqrt(
                p_spectrum_x_2**2+p_spectrum_y_2**2+p_spectrum_z_2**2)

            max_sweep_2 = np.max([x[500:999], y[500:999], z[500:999]])
            min_sweep_2 = np.min([x[500:999], y[500:999], z[500:999]])
            max_power_2 = np.max(
                [p_spectrum_x_2, p_spectrum_y_2, p_spectrum_z_2])

            p_spectrum_x_3, t_freq_x_3 = tremor_fft(x[1000:1499])
            p_spectrum_y_3, t_freq_y_3 = tremor_fft(y[1000:1499])
            p_spectrum_z_3, t_freq_z_3 = tremor_fft(z[1000:1499])
            p_spectrum_3_norm = np.sqrt(
                p_spectrum_x_3**2+p_spectrum_y_3**2+p_spectrum_z_3**2)

            max_sweep_3 = np.max([x[1000:1499], y[1000:1499], z[1000:1499]])
            min_sweep_3 = np.min([x[1000:1499], y[1000:1499], z[1000:1499]])
            max_power_3 = np.max(
                [p_spectrum_x_3, p_spectrum_y_3, p_spectrum_z_3])

            p_spectrum_x_4, t_freq_x_4 = tremor_fft(x[1500:1999])
            p_spectrum_y_4, t_freq_y_4 = tremor_fft(y[1500:1999])
            p_spectrum_z_4, t_freq_z_4 = tremor_fft(z[1500:1999])
            p_spectrum_4_norm = np.sqrt(
                p_spectrum_x_4**2+p_spectrum_y_4**2+p_spectrum_z_4**2)

            max_sweep_4 = np.max([x[1500:1999], y[1500:1999], z[1500:1999]])
            min_sweep_4 = np.min([x[1500:1999], y[1500:1999], z[1500:1999]])
            max_power_4 = np.max(
                [p_spectrum_x_4, p_spectrum_y_4, p_spectrum_z_4])

            p_spectrum_x_5, t_freq_x_5 = tremor_fft(x[2000:2499])
            p_spectrum_y_5, t_freq_y_5 = tremor_fft(y[2000:2499])
            p_spectrum_z_5, t_freq_z_5 = tremor_fft(z[2000:2499])
            p_spectrum_5_norm = np.sqrt(
                p_spectrum_x_5**2+p_spectrum_y_5**2+p_spectrum_z_5**2)

            max_sweep_5 = np.max([x[2000:2499], y[2000:2499], z[2000:2499]])
            min_sweep_5 = np.min([x[2000:2499], y[2000:2499], z[2000:2499]])
            max_power_5 = np.max(
                [p_spectrum_x_5, p_spectrum_y_5, p_spectrum_z_5])
            p_spectrum = np.mean([p_spectrum_1_norm, p_spectrum_2_norm,
                                 p_spectrum_3_norm, p_spectrum_4_norm, p_spectrum_5_norm], axis=0)

            f = 0
            for i in t_freq_x_1:
                if i > 4:
                    break
                f = f + 1
            h = 0
            for i in t_freq_x_1:
                if i > 14:
                    break
                h = h + 1
            c = 0
            peak_spectrum = np.max(p_spectrum[f:len(p_spectrum)-1])
            for i in p_spectrum:
                if i == peak_spectrum:
                    peak_freq = t_freq_x_1[c]
                    break
                c = c + 1

            total_power = np.sum(p_spectrum[f:h])
            for i in range(h-f):
                if np.sum(p_spectrum[f:f+i]) >= total_power/2:
                    print(i)
                    m_freq = t_freq_x_1[f+i]
                    break

            rms_sweep_1_x = rms_amplitude(x[0:499])
            rms_sweep_2_x = rms_amplitude(x[500:999])
            rms_sweep_3_x = rms_amplitude(x[1000:1499])
            rms_sweep_4_x = rms_amplitude(x[1500:1999])
            rms_sweep_5_x = rms_amplitude(x[2000:2499])
            rms_x = np.mean([rms_sweep_1_x, rms_sweep_2_x,
                            rms_sweep_3_x, rms_sweep_4_x, rms_sweep_5_x])

            rms_sweep_1_y = rms_amplitude(y[0:499])
            rms_sweep_2_y = rms_amplitude(y[500:999])
            rms_sweep_3_y = rms_amplitude(y[1000:1499])
            rms_sweep_4_y = rms_amplitude(y[1500:1999])
            rms_sweep_5_y = rms_amplitude(y[2000:2499])
            rms_y = np.mean([rms_sweep_1_y, rms_sweep_2_y,
                            rms_sweep_3_y, rms_sweep_4_y, rms_sweep_5_y])

            rms_sweep_1_z = rms_amplitude(z[0:499])
            rms_sweep_2_z = rms_amplitude(z[500:999])
            rms_sweep_3_z = rms_amplitude(z[1000:1499])
            rms_sweep_4_z = rms_amplitude(z[1500:1999])
            rms_sweep_5_z = rms_amplitude(z[2000:2499])
            rms_z = np.mean([rms_sweep_1_z, rms_sweep_2_z,
                            rms_sweep_3_z, rms_sweep_4_z, rms_sweep_5_z])

            apEn_sweep_1_x = approximate_entropy2(x[0:499], 2, 0.2)
            apEn_sweep_2_x = approximate_entropy2(x[500:999], 2, 0.2)
            apEn_sweep_3_x = approximate_entropy2(x[1000:1499], 2, 0.2)
            apEn_sweep_4_x = approximate_entropy2(x[1500:1999], 2, 0.2)
            apEn_sweep_5_x = approximate_entropy2(x[2000:2499], 2, 0.2)
            apEn_x = np.mean([apEn_sweep_1_x, apEn_sweep_2_x,
                              apEn_sweep_3_x, apEn_sweep_4_x, apEn_sweep_5_x])

            apEn_sweep_1_y = approximate_entropy2(y[0:499], 2, 0.2)
            apEn_sweep_2_y = approximate_entropy2(y[500:999], 2, 0.2)
            apEn_sweep_3_y = approximate_entropy2(y[1000:1499], 2, 0.2)
            apEn_sweep_4_y = approximate_entropy2(y[1500:1999], 2, 0.2)
            apEn_sweep_5_y = approximate_entropy2(y[2000:2499], 2, 0.2)
            apEn_y = np.mean([apEn_sweep_1_y, apEn_sweep_2_y,
                              apEn_sweep_3_y, apEn_sweep_4_y, apEn_sweep_5_y])

            apEn_sweep_1_z = approximate_entropy2(z[0:499], 2, 0.2)
            apEn_sweep_2_z = approximate_entropy2(z[500:999], 2, 0.2)
            apEn_sweep_3_z = approximate_entropy2(z[1000:1499], 2, 0.2)
            apEn_sweep_4_z = approximate_entropy2(z[1500:1999], 2, 0.2)
            apEn_sweep_5_z = approximate_entropy2(z[2000:2499], 2, 0.2)
            apEn_z = np.mean([apEn_sweep_1_z, apEn_sweep_2_z,
                              apEn_sweep_3_z, apEn_sweep_4_z, apEn_sweep_5_z])

            with t1:
                plt.figure(figsize=(5, 5))
                plt.plot(t, x, 'r')
                plt.plot(t, y, 'g')
                plt.plot(t, z, 'b')
                plt.xlabel("Tempo (s)")
                plt.ylabel("Aceleração (g)")
                plt.ylim(min_sweep_1, max_sweep_1)
                buffertremor1 = BytesIO()
                plt.savefig(buffertremor1, format="png")
                buffertremor1.seek(0)
                st.pyplot(plt)

            with t2:
                # avg
                plt.figure(figsize=(5, 5))
                plt.plot(t_freq_z_5, p_spectrum, 'k')
                plt.plot(peak_freq, peak_spectrum, marker='o', markersize=12,
                         markerfacecolor='none', markeredgecolor='r')
                plt.plot([m_freq, m_freq], [0, np.max(p_spectrum)*1.5], '--b')
                plt.xlabel("Frequência temporal (Hz)")
                plt.ylabel("Magnitude da aceleração (g)")
                plt.xlim(0, 14)
                plt.ylim(0, max_power_5)
                buffertremor2 = BytesIO()
                plt.savefig(buffertremor2, format="png")
                buffertremor2.seek(0)
                st.pyplot(plt)
            with t3:
                st.markdown(
                    '<h3 style="text-align: left; color: blue;">Resultados</h3>', unsafe_allow_html=True)
                st.text('Amplitude rms X (g) = ' + str(round(rms_x, 4)))
                st.text('Amplitude rms Y (g) = ' + str(round(rms_y, 4)))
                st.text('Amplitude rms Z (g) = ' + str(round(rms_z, 4)))
                st.text('Entropia aproximada X = ' + str(round(apEn_x, 3)))
                st.text('Entropia aproximada Y = ' + str(round(apEn_y, 3)))
                st.text('Entropia aproximada Z = ' + str(round(apEn_z, 3)))
                st.text('Amplitude de pico (g) = ' +
                        str(round(peak_spectrum, 3)))
                st.text('Frequência de pico (Hz) = ' +
                        str(round(peak_freq, 3)))
                st.text('Frequência mediana (Hz) = ' + str(round(m_freq, 3)))
                buffer = BytesIO()

                pdf = canvas.Canvas(buffer)

                # criando a linha
                pdf.setStrokeColorRGB(0, 0, 1)
                x1, y1 = 0, 769
                x2, y2 = 700, 769
                pdf.line(x1, y1, x2, y2)

                # criando o retângulo
                pdf.setFillColorRGB(0.8, 0.8, 1)  # Blue in RGB format
                pdf.setStrokeColorRGB(0, 0, 0, 0)  # Transparent in RGB format
                x1, y1 = 0, 0
                x2, y2 = 150, 900
                pdf.rect(x1, y1, x2 - x1, y2 - y1, fill=1)

                # criando texto de cabeçalho
                pdf.setFillColorRGB(0, 0, 1)  # Blue in RGB format
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(5, 800, "Universidade Federal do Pará")
                pdf.setFont("Helvetica", 8)
                pdf.drawString(5, 790, "Núcleo de Medicina Tropical")
                pdf.drawString(5, 780, "Laboratório de Neurologia Tropical")
                pdf.drawString(5, 770, "Atendimento em Saúde Digital")
                pdf.drawString(5, 720, "Smartphone: Xiaomi Mi 11 lite")
                pdf.drawString(5, 710, f"Mão testada: {hand}")
                pdf.drawString(5, 700, f"Condição {condition}")
                pdf.drawString(5, 690, "Tremor de mão")

                pdf.drawString(5, 100, "Núcleo de Medicina Tropical")
                pdf.drawString(5, 90, "Av. Generalíssimo Deodoro 92")
                pdf.drawString(5, 80, "Umarizal, Belém, Pará")
                pdf.drawString(5, 70, "66055240, Brasil")
                pdf.drawString(5, 60, "givagosouza@ufpa.br")
                pdf.drawString(5, 50, "91982653131")
                pdf.drawString(480, 20, "Givago da Silva Souza")
                pdf.drawString(470, 10, "Universidade Federal do Pará")

                pdf.setFillColorRGB(0, 0, 1)  # Blue in RGB format
                pdf.setFont("Helvetica-Bold", 14)
                pdf.drawString(
                    170, 780, "RELATÓRIO DE AVALIAÇÃO DO TREMOR DE MÃO")
                pdf.setFont("Helvetica-Bold", 8)
                pdf.drawString(155, 760, "Nome:")
                pdf.drawString(155, 745, "Data do teste:")
                pdf.drawString(155, 730, "Encaminhamento:")
                pdf.drawString(400, 760, "Data de nascimento:")
                pdf.drawString(400, 745, "Sexo:")
                pdf.drawString(400, 730, "Contato:")

                pdf.setFillColorRGB(0, 0, 0)
                pdf.setFont("Helvetica", 8)
                pdf.drawString(185, 760, name_participant)
                pdf.drawString(210, 745, str(date_participant))
                pdf.drawString(230, 730, doctor_participant)
                pdf.drawString(480, 760, str(birthdate_participant))
                pdf.drawString(425, 745, sex_participant)
                pdf.drawString(440, 730, contact_participant)

                pdf.setFillColorRGB(0, 0, 1)
                pdf.setFont("Helvetica-Bold", 12)
                pdf.drawString(160, 700, "Resultados:")

                pdf.setFont("Helvetica", 8)
                pdf.setFillColorRGB(0, 0, 0)
                pdf.drawString(
                    165, 680, 'Amplitude rms X (g) = ' + str(round(rms_x, 4)))
                pdf.drawString(
                    165, 660, 'Amplitude rms Y (g) = ' + str(round(rms_y, 4)))
                pdf.drawString(
                    165, 640, 'Amplitude rms Z (g) = ' + str(round(rms_z, 4)))
                pdf.drawString(
                    165, 620, 'Entropia aproximada X = ' + str(round(apEn_x, 4)))
                pdf.drawString(
                    165, 600, 'Entropia aproximada Y = ' + str(round(apEn_y, 4)))
                pdf.drawString(
                    165, 580, 'Entropia aproximada Z = ' + str(round(apEn_z, 4)))
                pdf.drawString(
                    165, 560, 'Amplitude de pico (g) = ' + str(round(peak_spectrum, 3)))
                pdf.drawString(
                    165, 540, 'Frequência de pico (Hz) = ' + str(round(peak_freq, 3)))
                pdf.drawString(
                    165, 520, 'Frequência mediana (Hz) = ' + str(round(m_freq, 3)))

                pdf.drawImage(ImageReader(buffertremor1), 180,
                              200, width=220, height=200)
                pdf.drawImage(ImageReader(buffertremor2), 380,
                              200, width=220, height=200)

                # Save the PDF
                pdf.save()
                buffer.seek(0)
                st.download_button("Criar relatório - Tremor", buffer)
                # Nome do arquivo de texto de saída
                output_file = "output.txt"

                # Abrir o arquivo para escrita
                with open(output_file, "w") as file:
                    file.write('Amplitude rms X (g) = ' +
                               str(round(rms_x, 4)) + "\n")
                    file.write('Amplitude rms Y (g) = ' +
                               str(round(rms_y, 4)) + "\n")
                    file.write('Amplitude rms Z (g) = ' +
                               str(round(rms_z, 4)) + "\n")
                    file.write('Entropia aproximada X = ' +
                               str(round(apEn_x, 4)) + "\n")
                    file.write('Entropia aproximada Y = ' +
                               str(round(apEn_y, 4)) + "\n")
                    file.write('Entropia aproximada Z = ' +
                               str(round(apEn_z, 4)) + "\n")
                    file.write('Amplitude de pico (g) = ' +
                               str(round(peak_spectrum, 3)) + "\n")
                    file.write('Frequência de pico (Hz) = ' +
                               str(round(peak_freq, 3)) + "\n")
                    file.write('Frequência mediana (Hz) = ' +
                               str(round(m_freq, 3)) + "\n")
                with open(output_file, "r") as file:
                    file_contents = file.read()

                # Usar st.download_button para baixar o arquivo
                st.download_button("Baixar resultados - Tremor",
                                   data=file_contents, key='download_Tremor_results')

with videos:
    st.markdown(
        '<h2 style="text-align: center; color: blue;">Tutorial para uso das rotinas</h2>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns([1, 1, 1])
    with t1:
        st.markdown(
            '<h3 style="text-align: center; color: blue;">Avaliação do equilíbrio estático</h3>', unsafe_allow_html=True)
        # Video file path or URL
        video_url = "https://www.youtube.com/watch?v=Xd1JNkeUy1M"

        # Display the video
        st.video(video_url)
        st.markdown(
            '<h3 style="text-align: center; color: blue;">Avaliação da mobilidade</h3>', unsafe_allow_html=True)
        # Video file path or URL
        video_url = "https://www.youtube.com/watch?v=Xd1JNkeUy1M"

        # Display the video
        st.video(video_url)
    with t2:
        st.markdown(
            '<h3 style="text-align: center; color: blue;">Tutorial para uso das rotinas</h3>', unsafe_allow_html=True)
        # Video file path or URL
        video_url = "https://www.youtube.com/watch?v=Xd1JNkeUy1M"

        # Display the video
        st.video(video_url)

        st.markdown(
            '<h3 style="text-align: center; color: blue;">Avaliação do tremor de mão</h3>', unsafe_allow_html=True)
        # Video file path or URL
        video_url = "https://www.youtube.com/watch?v=Xd1JNkeUy1M"

        # Display the video
        st.video(video_url)
