
NOTES
========

.. raw:: html


    <h2 id="trajectory-format">Trajectory Format</h2>
    <p>Benefits of using Numpy Z compression (npz) formatting: Standardized library that comes with installation of anaconda, Writes a compact file of several arrays into binary format which has a significantly smaller size over all other formats.</p>
    <table>
    <caption>Trajectory Format Comparison (writing)</caption>
    <tbody>
    <tr class="odd">
    <td style="text-align: center;"><strong>Process</strong></td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;"></td>
    </tr>
    <tr class="even">
    <td style="text-align: center;">(steps, states, lambda)</td>
    <td style="text-align: center;"><strong>H5</strong></td>
    <td style="text-align: center;"><strong>YAML</strong></td>
    <td style="text-align: center;"><strong>Numpy Z Compression</strong></td>
    <td style="text-align: center;"><strong>Python Pickle</strong></td>
    </tr>
    <tr class="odd">
    <td style="text-align: center;">1 M, 1000, 1</td>
    <td style="text-align: center;">3.57 min</td>
    <td style="text-align: center;">3.68 min</td>
    <td style="text-align: center;">3.50 min</td>
    <td style="text-align: center;">3.70 min</td>
    </tr>
    <tr class="even">
    <td style="text-align: center;">10 M, 50, 2</td>
    <td style="text-align: center;">31.53 min</td>
    <td style="text-align: center;">34.86 min</td>
    <td style="text-align: center;">31.43 min</td>
    <td style="text-align: center;">31.68 min</td>
    </tr>
    <tr class="odd">
    <td style="text-align: center;">Avg. size (MB)</td>
    <td style="text-align: center;">9.67</td>
    <td style="text-align: center;">11.11</td>
    <td style="text-align: center;">2.29</td>
    <td style="text-align: center;">9.69</td>
    </tr>
    </tbody>
    </table>
    <p>Processor 2.5 GHz Intel Core i5<br />
    </p>



