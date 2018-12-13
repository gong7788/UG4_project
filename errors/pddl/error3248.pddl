(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(clear b7)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b6 b9)
		(in-tower b6)
		(on b8 b6)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(blue b0)
		(maroon b0)
		(green b1)
		(maroon b1)
		(blue b2)
		(maroon b2)
		(green b3)
		(maroon b3)
		(blue b4)
		(maroon b4)
		(red b5)
		(blue b5)
		(green b5)
		(maroon b5)
		(red b6)
		(blue b6)
		(green b6)
		(maroon b6)
		(maroon b7)
		(maroon b8)
		(blue b9)
		(green b9)
		(maroon b9)
		(blue b7)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b8 b9)) (not (on b7 b9)) (not (on b5 b7)))))
)