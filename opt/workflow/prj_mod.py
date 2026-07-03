import ogstools as ot

#--------------------------------------------------
#prj updater function

def temp_prj(prj_in, prj_out,factors): 

    keff=factors['keff']

    values_str = " ".join(map(str, keff))

    model = ot.Project(input_file=prj_in,output_file=prj_out)
    xpath='./curves/curve[name="k_curve"]/values'
    medium=1

    try:
        model.replace_text(values_str,xpath)

    except Exception as e:
        print(f"CRITICAL ERROR in PRJ update: {e}")
        raise # Stop the loop if the input file is not correctly updated

    model.write_input()